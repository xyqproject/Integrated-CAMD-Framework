from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QPlainTextEdit, QWidget, QTextBrowser, QTextEdit, \
    QComboBox, QTableWidget, QTableWidgetItem
from PyQt6.QtWidgets import QCheckBox, QAbstractItemView, QLineEdit, QHBoxLayout, QHeaderView, QDialog, QVBoxLayout, \
    QLabel, QTabWidget, QMessageBox
from PyQt6 import uic
from PyQt6.QtCore import Qt, QRegularExpression
import sys
import pandas as pd
import functions_base
from PyQt6.QtGui import QFont, QColor, QBrush, QPalette, QRegularExpressionValidator
import os
import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget, QHeaderView, QLabel, QPushButton
)
from PyQt6.QtCore import Qt, QPoint, QRect
from PyQt6.QtGui import QMouseEvent, QPainter, QColor, QBrush
import re


class BubbleWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.ToolTip)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.label = QLabel(self)
        self.label.setStyleSheet("""
            QLabel {
                background-color: #f8f9fa;
                color: #333333;
                border: 1px solid #cccccc;
                border-radius: 6px;
                padding: 8px;
                font-size: 14px;
                max-width: 200px;
            }
        """)
        layout = QVBoxLayout(self)
        layout.addWidget(self.label)
        self.setLayout(layout)

    def set_text(self, text):
        self.label.setText(text)
        self.adjustSize()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRect(0, 0, self.width(), self.height())
        painter.setBrush(QBrush(QColor(248, 249, 250)))
        painter.setPen(QColor(204, 204, 204))
        painter.drawRoundedRect(rect, 6, 6)


class ClickableTable(QTableWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_widget = parent
        self.bubble = BubbleWidget(self)
        self.bubble.hide()
        self.cellClicked.connect(self.show_bubble_on_cell_click)

    def mousePressEvent(self, event: QMouseEvent):
        click_pos = event.pos()
        row = self.rowAt(click_pos.y())
        col = self.columnAt(click_pos.x())
        if row == -1 or col == -1:
            self.bubble.hide()
        else:
            super().mousePressEvent(event)

    def show_bubble_on_cell_click(self, row: int, col: int):
        cell_rect = self.visualItemRect(self.item(row, col))
        bubble_pos = self.mapToGlobal(QPoint(
            cell_rect.x() + cell_rect.width() + 10,
            cell_rect.y() + cell_rect.height() // 2
        ))
        detail_text = self.get_cell_detail(row, col)
        self.bubble.set_text(detail_text)
        self.bubble.move(bubble_pos)
        self.bubble.show()

    def get_cell_detail(self, row: int, col: int) -> str:
        cell_item = self.item(row, col)
        cell_text = cell_item.text() if cell_item else "空值"
        column_header_item = self.horizontalHeaderItem(col)
        column_name = column_header_item.text()
        if column_name in ["Molecule No", "Molecule1 No", "Molecule2 No"]:
            df_generated_molecule = pd.read_excel(file_path_generated_molecules, index_col=0)
            structural_text = ""
            for j in df_generated_molecule.columns:
                if df_generated_molecule.loc[float(cell_text) - 1, j] > 0:
                    structural_text += f"{j}: {df_generated_molecule.loc[float(cell_text) - 1, j]}\n"
            return structural_text
        else:
            return cell_text


def format_3sig(value):
    try:
        num = float(value)
        return f"{num:.3g}"
    except:
        return str(value)


def Molecular_Generation(table_generated_molecules):
    list_group = []
    row1 = table_group_selected.rowCount()
    if row1 == 0:
        QMessageBox.information(ui, 'Notice', 'There is no selected group!')
        return
    for i in range(row1):
        list_group.append(table_group_selected.item(i, 0).text())
    print(list_group)
    list_structural_constraints = {}
    list_structural_constraints["Molecular Weight"] = (float(mw_lb.text()), float(mw_ub.text()))
    list_structural_constraints["The same groups"] = (float(g_lb.text()), float(g_ub.text()))
    list_structural_constraints["All groups"] = (float(al_lb.text()), float(al_ub.text()))
    list_structural_constraints["All functional groups"] = (float(alf_lb.text()), float(alf_ub.text()))
    list_structural_constraints["q"] = (float(q_lb.text()), float(q_ub.text()))
    try:
        res = functions_base.molecular_generation(df_group, df_valency, df_weight, list_group, list_structural_constraints)
    except Exception as e:
        QMessageBox.critical(ui, 'Error', f'Molecular generation failed:\n{e}')
        return
    if res.empty:
        table_generated_molecules.setRowCount(0)
        QMessageBox.information(ui, 'Notice', 'There is no feasible molecules!')
        return
    res.to_excel(file_path_generated_molecules)
    res.index += 1
    table_generated_molecules.setRowCount(0)
    for i in range(len(res.index)):
        table_generated_molecules.insertRow(i)
        data = QTableWidgetItem(format_3sig(res.index[i]))
        data.setTextAlignment(Qt.AlignmentFlag.AlignHCenter)
        table_generated_molecules.setItem(i, 0, data)
        for j in range(row1):
            data = QTableWidgetItem(format_3sig(res.iloc[i, j]))
            data.setTextAlignment(Qt.AlignmentFlag.AlignHCenter)
            table_generated_molecules.setItem(i, j + 1, data)


def target_select(checked, row, table_target_selected, table_model):
    if checked:
        set_selected_target.add(row)
    else:
        if row in set_selected_target:
            set_selected_target.remove(row)
    update_selected_target(table_target_selected, table_model)


def target_select_cop(checked, row, table_target_selected, table_model, checkbox_item):
    if checked:
        set_selected_target.add(row)
        global current_row
        current_row = row
        show_input_dialog()
    else:
        if row in set_selected_target:
            set_selected_target.remove(row)
        global T_evaporating
        global T_condensing
        T_condensing = -1
        T_evaporating = -1
    update_selected_target(table_target_selected, table_model)


def show_input_dialog():
    global current_row
    global T_Cooling
    global T_Refrigeration
    global Stage_Number
    if current_row == -1:
        return
    dialog = QDialog()
    dialog.setWindowTitle("Input the process constraints")
    dialog.resize(300, 150)
    value1 = ""
    value2 = ""
    value3 = ""
    layout = QVBoxLayout()
    hbox1 = QHBoxLayout()
    hbox1.addWidget(QLabel("Input cooling temperature:"))
    line_edit1 = QLineEdit()
    line_edit1.setText("313.15")
    hbox1.addWidget(line_edit1)
    layout.addLayout(hbox1)
    hbox2 = QHBoxLayout()
    hbox2.addWidget(QLabel("Input refrigeration temperature:"))
    line_edit2 = QLineEdit()
    line_edit2.setText("260.15")
    hbox2.addWidget(line_edit2)
    layout.addLayout(hbox2)
    hbox3 = QHBoxLayout()
    hbox3.addWidget(QLabel("Input the number of stage:"))
    line_edit3 = QLineEdit()
    line_edit3.setText("1")
    regex = QRegularExpression("^[12]$")
    validator = QRegularExpressionValidator(regex, line_edit3)
    line_edit3.setValidator(validator)
    hbox3.addWidget(line_edit3)
    layout.addLayout(hbox3)
    btn_layout = QHBoxLayout()

    def on_accept():
        nonlocal value1, value2, value3
        value1 = line_edit1.text()
        value2 = line_edit2.text()
        value3 = line_edit3.text()
        dialog.accept()

    ok_btn = QPushButton("OK")
    ok_btn.clicked.connect(on_accept)
    cancel_btn = QPushButton("Cancel")
    cancel_btn.clicked.connect(dialog.reject)
    btn_layout.addWidget(ok_btn)
    btn_layout.addWidget(cancel_btn)
    layout.addLayout(btn_layout)
    dialog.setLayout(layout)
    result = dialog.exec()
    if result == QDialog.DialogCode.Accepted:
        T_Cooling = float(value1)
        T_Refrigeration = float(value2)
        Stage_Number = float(value3)
        print(f"T_Cooling={T_Cooling}, T_Refrigeration={T_Refrigeration}, Stage_Number={Stage_Number}")
    current_row = -1


def update_selected_target(table_target_selected, table_model):
    table_target_selected.setRowCount(0)
    table_model.setRowCount(0)
    for row in sorted(set_selected_target):
        property_selected = table_target.item(row, 2).text()
        new_row = table_target_selected.rowCount()
        table_target_selected.insertRow(new_row)
        name_item = QTableWidgetItem(property_selected)
        name_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        table_target_selected.setItem(new_row, 0, name_item)
        lowerbound_item = QTableWidgetItem("-99999")
        lowerbound_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        table_target_selected.setItem(new_row, 1, lowerbound_item)
        upperbound_item = QTableWidgetItem("99999")
        upperbound_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        table_target_selected.setItem(new_row, 2, upperbound_item)
        checkbox = QCheckBox()
        checkbox.toggled.connect(lambda checked, r=new_row: on_toggled(r, checked))
        cell_widget = QWidget()
        layout = QHBoxLayout(cell_widget)
        layout.addWidget(checkbox, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.setContentsMargins(0, 0, 0, 0)
        table_target_selected.setCellWidget(new_row, 3, cell_widget)
        property_selected = table_target.item(row, 2).text()
        if property_selected != "Coefficient of Performance":
            new_row = table_model.rowCount()
            table_model.insertRow(new_row)
            name_item = QTableWidgetItem(property_selected)
            name_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            table_model.setItem(new_row, 0, name_item)
            combo = QComboBox()
            moedl_list = ["Linear Model", "Support Vector Regression", "Guassian Process Regression"]
            combo.addItems(moedl_list)
            combo.setCurrentIndex(0)
            table_model_selected[property_selected] = moedl_list[0]
            combo.currentIndexChanged.connect(
                lambda index, name=property_selected, opts=moedl_list: model_type_combobox_changed(name, opts, index))
            table_model.setCellWidget(new_row, 1, combo)
    set_table_color(table_target_selected)
    set_table_color(table_model)


def on_toggled(row, checked):
    global selected_objective_No
    if checked:
        if selected_objective_No != -1:
            prev_cell = table_target_selected.cellWidget(selected_objective_No, 3)
            if prev_cell:
                for widget in prev_cell.findChildren(QCheckBox):
                    widget.blockSignals(True)
                    widget.setChecked(False)
                    widget.blockSignals(False)
        selected_objective_No = row
    else:
        if selected_objective_No == row:
            selected_objective_No = -1
    print(selected_objective_No)


def model_type_combobox_changed(name, opts, index):
    table_model_selected[name] = opts[index]


def group_select(checked, row, table_group_selected, table_generated_molecules):
    if checked:
        set_selected_group.add(row)
    else:
        if row in set_selected_group:
            set_selected_group.remove(row)
    update_selected_group(table_group_selected, table_generated_molecules)


def update_selected_group(table_group_selected, table_generated_molecules):
    table_group_selected.setRowCount(0)
    for row in sorted(set_selected_group):
        group_selected = table_group.item(row, 1).text()
        new_row = table_group_selected.rowCount()
        table_group_selected.insertRow(new_row)
        name_item = QTableWidgetItem(group_selected)
        name_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        table_group_selected.setItem(new_row, 0, name_item)
    table_generated_molecules.setColumnCount((1 + len(set_selected_group)))
    table_generated_molecules.setHorizontalHeaderLabels(
        ["Molecule No"] + [df_group.iloc[i, 0] for i in set_selected_group])
    table_generated_molecules.resizeColumnsToContents()
    table_generated_molecules.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Fixed)


def Property_Prediction(table_predicted_Property):
    df_step1_out = pd.read_excel(file_path_generated_molecules, index_col=0)
    df_all_groups = pd.read_excel(file_path_all_groups, index_col=None, header=0)
    lst_property_model = set()
    for property, model_type in table_model_selected.items():
        lst_property_model.add((df_property_symbol.loc[property, "symbol"], model_type))
    print(lst_property_model)
    res = functions_base.property_prediction(df_step1_out, df_all_groups, lst_property_model)
    res.to_excel(file_path_property_prediction)
    table_predicted_Property.setRowCount(0)
    table_predicted_Property.setColumnCount((1 + len(res.columns)))
    headers = ["Molecule No"] + list(map(str, res.columns))
    table_predicted_Property.setHorizontalHeaderLabels(headers)
    for i in range(len(res.index)):
        table_predicted_Property.insertRow(i)
        data = QTableWidgetItem(str(res.index[i]))
        data.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        table_predicted_Property.setItem(i, 0, data)
        for j in range(len(res.columns)):
            data = QTableWidgetItem(format_3sig(res.iloc[i, j]))
            data.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            table_predicted_Property.setItem(i, j + 1, data)
    if table_predicted_Property.columnCount() != 0:
        table_width = table_predicted_Property.viewport().width()
        col_count = table_predicted_Property.columnCount()
        base_width = table_width // col_count
        remaining = table_width % col_count
        for col in range(col_count):
            width = base_width + (1 if col < remaining else 0)
            table_predicted_Property.setColumnWidth(col, width)


def Molecular_Screen(table_optimal_molecule):
    sol_num = int(float(number_solution.text()))
    df_molecular_list = pd.read_excel(file_path_property_prediction, index_col=0)
    list_property_constraints = {}
    row2 = table_target_selected.rowCount()
    for i in range(row2):
        if i != selected_objective_No:
            p, lb, ub = (table_target_selected.item(i, j).text() for j in range(3))
            list_property_constraints[df_property_symbol.loc[p, "symbol"]] = (int(lb), int(ub))
    property_objective = df_property_symbol.loc[table_target_selected.item(selected_objective_No, 0).text(), "symbol"]
    if property_objective == "cop":
        if Stage_Number == 1:
            table_optimal_molecule.setRowCount(0)
            table_optimal_molecule.setColumnCount(13)
            headers = ["Molecule No", "COP", "tb", "R(tb)", "tc", "R(tc)", "pc", "R(pc)", "Ag", "Bg", "Cg", "Dg",
                       "R(cpg)"]
            table_optimal_molecule.setHorizontalHeaderLabels(headers)
            res = functions_base.molecular_screen_consider_reliability_cop_1_stage(df_molecular_list,
                                                                                   list_property_constraints, T_Cooling,
                                                                                   T_Refrigeration, sol_num)
            res.to_excel(file_path_solutions)
            if res.dropna(how='all').empty:
                QMessageBox.information(ui, 'Notice', 'There is no feasible solution!')
                return
            for j in res.index:
                new_row = table_optimal_molecule.rowCount()
                table_optimal_molecule.insertRow(new_row)
                for i in range(len(headers)):
                    item = QTableWidgetItem(format_3sig(res.iloc[j, i]))
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    table_optimal_molecule.setItem(j, i, item)
        if Stage_Number == 2:
            table_optimal_molecule.setRowCount(0)
            table_optimal_molecule.setColumnCount(25)
            headers = ["Molecule1 No", "Molecule2 No", "COP", "tb1", "R(tb1)", "tc1", "R(tc1)", "pc1", "R(pc1)", "Ag1",
                       "Bg1", "Cg1", "Dg1", "R(cpg1)", "tb2", "R(tb2)", "tc2", "R(tc2)", "pc2", "R(pc2)", "Ag2", "Bg2",
                       "Cg2", "Dg2", "R(cpg2)"]
            table_optimal_molecule.setHorizontalHeaderLabels(headers)
            res = functions_base.molecular_screen_consider_reliability_cop_2_stage(df_molecular_list,
                                                                                   list_property_constraints, T_Cooling,
                                                                                   T_Refrigeration, sol_num)
            res.to_excel(file_path_solutions)
            if res.dropna(how='all').empty:
                QMessageBox.information(ui, 'Notice', 'There is no feasible solution!')
                return
            for j in res.index:
                new_row = table_optimal_molecule.rowCount()
                table_optimal_molecule.insertRow(new_row)
                for i in range(len(headers)):
                    item = QTableWidgetItem(format_3sig(res.iloc[j, i]))
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    table_optimal_molecule.setItem(j, i, item)
    else:
        table_optimal_molecule.setRowCount(0)
        table_optimal_molecule.setColumnCount((3 + 2 * len(list_property_constraints.keys())))
        headers = ["Molecule No"] + [property_objective, f"R({property_objective})"] + [x for item in list(
            list_property_constraints.keys()) for x in (item, f"R({item})")]
        table_optimal_molecule.setHorizontalHeaderLabels(headers)
        res = functions_base.molecular_screen_consider_reliability_no_integrated_process(df_molecular_list,
                                                                                         list_property_constraints,
                                                                                         property_objective, sol_num)
        res.to_excel(file_path_solutions)
        if res.dropna(how='all').empty:
            QMessageBox.information(ui, 'Notice', 'There is no feasible solution!')
            return
        for j in res.index:
            new_row = table_optimal_molecule.rowCount()
            table_optimal_molecule.insertRow(new_row)
            for i in range(len(headers)):
                item = QTableWidgetItem(format_3sig(res.iloc[j, i]))
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                table_optimal_molecule.setItem(j, i, item)
    if table_optimal_molecule.columnCount() != 0:
        table_width = table_optimal_molecule.viewport().width()
        col_count = table_optimal_molecule.columnCount()
        base_width = table_width // col_count
        remaining = table_width % col_count
        table_optimal_molecule.resizeColumnsToContents()
        table_optimal_molecule.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Fixed)


def product_type_change(table_target):
    global selected_objective_No
    selected_objective_No = -1
    table_target_selected.setRowCount(0)
    table_model.setRowCount(0)
    set_selected_target.clear()
    file_path_target_product = os.path.join(f"data/stored/{select_product.currentText()}",
                                            f"Target for {select_product.currentText()}.xlsx")
    df_target_product = pd.read_excel(file_path_target_product, index_col=0)
    row = len(df_target_product.index)
    vol = len(df_target_product.columns)
    table_target.setRowCount(row)
    table_target.setColumnCount(vol)
    table_target.setHorizontalHeaderLabels(["Requirement", "Use", "TargetProperty"])
    for i in range(row):
        for j in [0, 2]:
            data = QTableWidgetItem(str(df_target_product.iloc[i, j]))
            data.setTextAlignment(Qt.AlignmentFlag.AlignHCenter)
            table_target.setItem(i, j, data)
        checkbox_item = QCheckBox()
        checkbox_item.setChecked(False)
        if df_target_product.iloc[i, 0] == "Efficiency of Refrigeration":
            checkbox_item.stateChanged.connect(
                lambda checked, r=i: target_select_cop(checked, r, table_target_selected, table_model, checkbox_item))
        else:
            checkbox_item.stateChanged.connect(
                lambda checked, r=i: target_select(checked, r, table_target_selected, table_model))
        cell_widget = QWidget()
        layout = QHBoxLayout(cell_widget)
        layout.addWidget(checkbox_item, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.setContentsMargins(0, 0, 0, 0)
        table_target.setCellWidget(i, 1, cell_widget)
    table_target.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    set_table_color(table_target)


def set_table_color(tablewidget):
    tablewidget.setShowGrid(True)
    tablewidget.setGridStyle(Qt.PenStyle.SolidLine)
    tablewidget.setAlternatingRowColors(True)
    font = QFont("Consolas", 10)
    tablewidget.setFont(font)
    tablewidget.resizeRowsToContents()
    tablewidget.setAlternatingRowColors(True)
    tablewidget.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    tablewidget.setStyleSheet("""
        QTableWidget {
            border: 1px solid #000000;
            border-radius: 1px;
            padding: 2px;
            background-color: white;
        }
        QHeaderView::section {
            background-color: #6c7aae;
            color: white;
            padding: 2px;
            border: 1px solid #000000;
            border-radius: 1px;
            text-align: center;
        }
        QTableWidget::item {
            text-align: center;
            border: 1px solid #000000;
            padding: 2px;
        }
        QTableWidget::item:selected {
            background-color: #a8b4e0;
            color: black;
        }
        QTableWidget::gridline {
            color: #000000;
        }

    """)
    tablewidget.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
    tablewidget.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
    tablewidget.setEditTriggers(QTableWidget.EditTrigger.DoubleClicked)
    tablewidget.setFocusPolicy(Qt.FocusPolicy.StrongFocus)


def set_lineedit_color(lineeditwidget):
    font = QFont("Consolas", 10)
    lineeditwidget.setFont(font)
    lineeditwidget.setMinimumHeight(24)
    lineeditwidget.setMaximumHeight(28)
    lineeditwidget.setStyleSheet("""
            #GrayLineEdit {
                background-color: #ffffff;
                color: #333333;
                border: 1px solid #b0b0b0;
                border-radius: 4px;
                padding: 6px 8px;
            }

            #GrayLineEdit:focus {
                border: 2px solid #909090;
                background-color: #fafafa;
            }

            #GrayLineEdit:disabled {
                background-color: #f0f0f0;
                color: #999999;
                border-color: #d0d0d0;
            }
     """)
    lineeditwidget.setClearButtonEnabled(True)
    lineeditwidget.setAlignment(Qt.AlignmentFlag.AlignLeft)


def set_combobox_color(comboboxwidget):
    font = QFont("Consolas", 10)
    comboboxwidget.setFont(font)
    comboboxwidget.setMinimumHeight(28)
    comboboxwidget.setMaximumHeight(32)
    comboboxwidget.setStyleSheet("""
        QComboBox {
            text-align: center;
            background-color: #f0f0f0;
            color: #333333;
            border: 1px solid #a0a0a0;
            padding: 5px 30px 5px 8px;
            selection-background-color: #3d7fe6;
        }
        QComboBox:focus {
            border: 1px solid #0066cc;
            background-color: #ffffff;
        }
        QComboBox:disabled {
            background-color: #e0e0e0;
            color: #888888;
            border: 1px solid #c0c0c0;
        }
        QComboBox::drop-down {
            subcontrol-origin: padding;
            subcontrol-position: top right;
            width: 25px;
            border-left: 1px solid #a0a0a0;
        }
        QComboBox::down-arrow {
            image: url(:/icons/down_arrow.png);
            width: 10px;
            height: 10px;
        }
        QComboBox QAbstractItemView {
            background-color: #f0f0f0;
            color: #333333;
            border: 1px solid #a0a0a0;
            selection-background-color: #3d7fe6;
            selection-color: white;
            padding: 2px;
            font-family: Consolas;
            font-size: 10pt;
        }
        QComboBox QAbstractItemView::item {
            height: 24px;
            padding: 2px 8px;
        }
        QComboBox QAbstractItemView::item:hover {
            background-color: #e0e0e0;
        }
    """)
    comboboxwidget.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    comboboxwidget.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
    comboboxwidget.setEditable(False)


def set_button_styles(buttonwidget):
    font = QFont("Consolas", 10)
    buttonwidget.setFont(font)
    buttonwidget.setMinimumHeight(35)
    buttonwidget.setMinimumWidth(150)
    buttonwidget.setMaximumHeight(50)
    buttonwidget.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    buttonwidget.setStyleSheet("""
            QPushButton {
                background-color: #34495e;
                color: #ecf0f1;
                border: 1px solid #2c3e50;
                border-radius: 2px;
                padding: 6px 12px;
                font-weight: bold;
            }

            QPushButton:hover {
                background-color: #3d5a80;
                border: 1px solid #547aa5;
            }

            QPushButton:pressed {
                background-color: #293949;
                border: 1px solid #1f2c38;
                padding-left: 7px;
                padding-top: 7px;
            }

            QPushButton:disabled {
                background-color: #2c3e50;
                color: #7f8c8d;
                border: 1px solid #34495e;
            }

            QPushButton#primaryBtn {
                background-color: #2980b9;
            }

            QPushButton#primaryBtn:hover {
                background-color: #3498db;
            }

            QPushButton#primaryBtn:pressed {
                background-color: #2471a3;
            }

            QPushButton#dangerBtn {
                background-color: #c0392b;
            }

            QPushButton#dangerBtn:hover {
                background-color: #e74c3c;
            }

            QPushButton#dangerBtn:pressed {
                background-color: #a52a1d;
            }
    """)


def set_mainwindow_styles(mainwindow):
    mainwindow.setWindowTitle("Sim-CAMD")
    mainwindow.setStyleSheet("""
        QMainWindow {
            background-color: #f0f0f0;
        }
        QToolBar {
            background-color: #e0e0e0;
            border-bottom: 1px solid #a0a0a0;
            spacing: 1px;
        }
        QToolButton {
            background-color: #e0e0e0;
            border: 1px solid #a0a0a0;
            padding: 5px;
            margin: 1px;
        }
        QToolButton:hover {
            background-color: #d0d0d0;
        }
        QToolButton:pressed {
            background-color: #b0b0b0;
        }
        QStatusBar {
            background-color: #e0e0e0;
            border-top: 1px solid #a0a0a0;
            color: #333333;
            font-family: Consolas;
            font-size: 9pt;
        }
        QDockWidget {
            titlebar-close-icon: none;
            titlebar-normal-icon: none;
            background-color: #f0f0f0;
            border: 1px solid #a0a0a0;
        }
        QDockWidget::title {
            background-color: #404040;
            color: white;
            padding: 5px;
            font-weight: bold;
        }
    """)


def set_tabwidget_styles(tabwidget):
    font = QFont("Consolas", 9)
    tabwidget.setFont(font)
    tabwidget.setStyleSheet("""
        QTabWidget::pane {
            border: 1px solid #a0a0a0;
            background-color: #f5f5f5;
            margin-top: 2px;
        }
        QTabBar {
            background-color: #e0e0e0;
            border-bottom: 1px solid #a0a0a0;
        }
        QTabBar::tab {
            background-color: #e0e0e0;
            color: #333333;
            padding: 6px 16px;
            border: 1px solid #a0a0a0;
            border-bottom-color: #a0a0a0;
            margin-right: 2px;
            font-weight: bold;
        }
        QTabBar::tab:selected {
            background-color: #f5f5f5;
            border-color: #a0a0a0;
            border-bottom-color: #f5f5f5;
            font-weight: bold;
        }
        QTabBar::tab:hover:!selected {
            background-color: #d0d0d0;
        }
        QTabBar::tab:disabled {
            color: #888888;
            background-color: #e5e5e5;
        }
    """)
    tabwidget.setElideMode(Qt.TextElideMode.ElideRight)
    tabwidget.setDocumentMode(False)
    tabwidget.setMovable(False)
    tabwidget.setTabShape(QTabWidget.TabShape.Rounded)


def set_textbrowser_styles(textbrowser):
    textbrowser.setStyleSheet("""
            QTextBrowser {
                background-color: white;
                color: #333;
                border: 1px solid #ccc;
                padding: 2px;
                font-family: "Consolas", sans-serif;
                font-size: 20px;
                line-height: 2.0;
                text-align: center;
                vertical-align: middle;
            }
            QScrollBar:vertical {
                background-color: #f0f0f0;
                width: 12px;
            }
            QScrollBar::handle:vertical {
                background-color: #bbb;
                min-height: 30px;
                border-radius: 6px;
            }
            QScrollBar::horizontal {
                background-color: #f0f0f0;
                height: 12px;
            }
            QScrollBar::handle:horizontal {
                background-color: #bbb;
                min-width: 30px;
                border-radius: 6px;
            }
        """)
    textbrowser.setAlignment(Qt.AlignmentFlag.AlignCenter)
    font = QFont("Consolas", 10)
    textbrowser.setFont(font)


def on_tab_changed(index):
    if index == 3:
        try:
            result_text = "Objective property:\n"
            if selected_objective_No != -1:
                property_objective = df_property_symbol.loc[
                    table_target_selected.item(selected_objective_No, 0).text(), "symbol"]
                result_text += f"            {property_objective}\n"
            result_text += "Constraints:\n"
            row2 = table_target_selected.rowCount()
            for i in range(row2):
                if i != selected_objective_No:
                    p, lb, ub = (table_target_selected.item(i, j).text() for j in range(3))
                    p_symbol = df_property_symbol.loc[p, "symbol"]
                    result_text += f"      {int(lb)}<={p_symbol}<={int(ub)}\n"
            summary_information.setText(result_text)
        except ValueError:
            summary_information.setText("Input error! please input suitable value!")


if __name__ == "__main__":
    file_path_generated_molecules = os.path.join("results/temp file", "generated_molecules.xlsx")
    file_path_all_groups = os.path.join("data/stored", "220group.xlsx")
    file_path_property_prediction = os.path.join("results/temp file", "property_prediction.xlsx")
    file_path_solutions = os.path.join("results/temp file", "solutions.xlsx")
    file_path_group_info = os.path.join("data/stored", "group_Noc_Mw.xlsx")
    file_path_property_symbol = os.path.join("data/stored", "property symbol.xlsx")
    file_path_UI = os.path.join("UI", "mainwindow_ui.ui")
    file_path_target_solvent = os.path.join("data/stored/solvent", "Target for solvent.xlsx")
    df_group = pd.read_excel(file_path_group_info, sheet_name="group", index_col=0)
    df_valency = pd.read_excel(file_path_group_info, sheet_name="valency", index_col=0)
    df_weight = pd.read_excel(file_path_group_info, sheet_name="weight", index_col=0)
    df_property_symbol = pd.read_excel(file_path_property_symbol, index_col=0)
    app = QApplication(sys.argv)
    ui = uic.loadUi(file_path_UI)
    select_product: QComboBox = ui.comboBox1
    select_product.addItems(["Solvent", "Refrigerant"])
    select_product.setCurrentIndex(0)
    select_product.currentIndexChanged.connect(lambda: product_type_change(table_target))
    set_selected_target = set()
    selected_objective_No = -1
    current_row = -1
    table_target: QTableWidget = ui.tableWidget1
    table_target.setStyleSheet("background-color: #f0f0f0;")
    df_target_solvent = pd.read_excel(file_path_target_solvent, index_col=0)
    row = len(df_target_solvent.index)
    vol = len(df_target_solvent.columns)
    table_target.setRowCount(row)
    table_target.setColumnCount(vol)
    table_target.setHorizontalHeaderLabels(["Requirement", "Use", "TargetProperty"])
    for i in range(row):
        for j in [0, 2]:
            data = QTableWidgetItem(str(df_target_solvent.iloc[i, j]))
            data.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            data.setFlags(data.flags() & ~Qt.ItemFlag.ItemIsEditable)
            table_target.setItem(i, j, data)
    table_target_selected: QTableWidget = ui.tableWidget2
    table_target_selected.setColumnCount(4)
    table_target_selected.setHorizontalHeaderLabels(["TargetProperty", "MinValue", "MaxValue", "Set as Objective"])
    table_model: QTableWidget = ui.tableWidget7
    table_model.setColumnCount(2)
    table_model.setHorizontalHeaderLabels(["TargetProperty", "Model Type"])
    for i in range(row):
        checkbox_item = QCheckBox()
        checkbox_item.setChecked(False)
        checkbox_item.stateChanged.connect(
            lambda checked, r=i: target_select(checked, r, table_target_selected, table_model))
        cell_widget = QWidget()
        layout = QHBoxLayout(cell_widget)
        layout.addWidget(checkbox_item, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.setContentsMargins(0, 0, 0, 0)
        table_target.setCellWidget(i, 1, cell_widget)
    set_selected_group = set()
    table_group: QTableWidget = ui.tableWidget3
    row = len(df_group.index)
    table_group.setRowCount(row)
    table_group.setColumnCount(4)
    table_group.setHorizontalHeaderLabels(["Select", "Group", "Valency", "Weight"])
    for i in range(row):
        data = QTableWidgetItem(str(df_group.iloc[i, 0]))
        data.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        table_group.setItem(i, 1, data)
        data = QTableWidgetItem(str(df_valency.iloc[i, 0]))
        data.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        table_group.setItem(i, 2, data)
        data = QTableWidgetItem(str(df_weight.iloc[i, 0]))
        data.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        table_group.setItem(i, 3, data)
    table_group.resizeColumnsToContents()
    table_group.resizeRowsToContents()
    table_group.setAlternatingRowColors(True)
    table_group.horizontalHeader().setStretchLastSection(True)
    table_group.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    table_group_selected: QTableWidget = ui.tableWidget4
    table_group_selected.setColumnCount(1)
    table_group_selected.setHorizontalHeaderLabels(["Group"])
    for i in range(row):
        checkbox_item = QCheckBox()
        checkbox_item.setChecked(False)
        checkbox_item.stateChanged.connect(
            lambda checked, r=i: group_select(checked, r, table_group_selected, table_generated_molecules))
        cell_widget = QWidget()
        layout = QHBoxLayout(cell_widget)
        layout.addWidget(checkbox_item, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.setContentsMargins(0, 0, 0, 0)
        table_group.setCellWidget(i, 0, cell_widget)
    table_group_selected.resizeColumnsToContents()
    table_group_selected.resizeRowsToContents()
    table_group_selected.setAlternatingRowColors(True)
    table_group_selected.horizontalHeader().setStretchLastSection(True)
    mw_lb: QLineEdit = ui.lineEdit_mw_lb
    mw_lb.setText("0")
    mw_ub: QLineEdit = ui.lineEdit_mw_ub
    mw_ub.setText("100")
    g_lb: QLineEdit = ui.lineEdit_g_lb
    g_lb.setText("0")
    g_ub: QLineEdit = ui.lineEdit_g_ub
    g_ub.setText("4")
    al_lb: QLineEdit = ui.lineEdit_al_lb
    al_lb.setText("2")
    al_ub: QLineEdit = ui.lineEdit_al_ub
    al_ub.setText("6")
    alf_lb: QLineEdit = ui.lineEdit_alf_lb
    alf_lb.setText("0")
    alf_ub: QLineEdit = ui.lineEdit_alf_ub
    alf_ub.setText("2")
    q_lb: QLineEdit = ui.lineEdit_q_lb
    q_lb.setText("1")
    q_ub: QLineEdit = ui.lineEdit_q_ub
    q_ub.setText("1")
    table_generated_molecules: QTableWidget = ui.tableWidget5
    table_model_selected = {}
    table_predicted_Property: QTableWidget = ui.tableWidget6
    ui.pushButton3.clicked.connect(lambda: Property_Prediction(table_predicted_Property))
    summary_information: QTextBrowser = ui.textBrowser1
    ui.tabWidget.currentChanged.connect(on_tab_changed)
    T_Cooling = -1
    T_Refrigeration = -1
    Stage_Number = 1
    number_solution: QLineEdit = ui.lineEdit_n_sol
    number_solution.setText("1")
    solution_widget: QWidget = ui.widget1
    main_layout = QVBoxLayout(solution_widget)
    main_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
    main_layout.setContentsMargins(0, 0, 0, 0)
    table_optimal_molecule = ClickableTable()
    table_optimal_molecule.setStyleSheet("background-color: white; border: 1px solid #eee;")
    table_optimal_molecule.setFixedHeight(450)
    table_optimal_molecule.setMinimumWidth(700)
    table_optimal_molecule.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    main_layout.addWidget(table_optimal_molecule, alignment=Qt.AlignmentFlag.AlignBottom)
    ui.pushButton4.clicked.connect(lambda: Molecular_Screen(table_optimal_molecule))
    set_table_color(table_target)
    table_target.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    set_table_color(table_target_selected)
    set_table_color(table_group)
    set_table_color(table_group_selected)
    set_table_color(table_generated_molecules)
    set_table_color(table_model)
    set_table_color(table_predicted_Property)
    set_table_color(table_optimal_molecule)
    set_lineedit_color(mw_lb)
    set_lineedit_color(mw_lb)
    set_lineedit_color(mw_ub)
    set_lineedit_color(g_lb)
    set_lineedit_color(g_ub)
    set_lineedit_color(al_lb)
    set_lineedit_color(al_ub)
    set_lineedit_color(alf_lb)
    set_lineedit_color(alf_ub)
    set_lineedit_color(q_lb)
    set_lineedit_color(q_ub)
    set_lineedit_color(number_solution)
    set_combobox_color(select_product)
    set_button_styles(ui.pushButton2)
    set_button_styles(ui.pushButton3)
    set_button_styles(ui.pushButton4)
    set_mainwindow_styles(ui)
    set_tabwidget_styles(ui.tabWidget)
    set_textbrowser_styles(summary_information)
    ui.pushButton2.clicked.connect(lambda: Molecular_Generation(table_generated_molecules))
    ui.show()
    sys.exit(app.exec())
