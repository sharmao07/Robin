APP_STYLESHEET = """
* { font-family: -apple-system, "SF Pro Display", sans-serif; }
QWidget { color: #E6EAF2; font-size: 13px; }
QMainWindow { background: #0A0C12; }
QWidget#Workspace { background: #0A0C12; }
QFrame#Sidebar { background: #10131B; border-right: 1px solid #242936; }
QLabel#BrandMark { background: #8B5CF6; color: #FFFFFF; border-radius: 12px; font-size: 21px; font-weight: 800; }
QLabel#BrandName { color: #F8FAFC; font-size: 22px; font-weight: 800; }
QLabel#Eyebrow { color: #81899B; font-size: 10px; font-weight: 800; letter-spacing: 1.4px; }
QLabel#MutedText { color: #929BAD; font-size: 12px; }
QLabel#PageTitle { color: #F5F7FB; font-size: 25px; font-weight: 750; }
QLabel#StatusPill { background: #14271F; color: #7DE2AD; border: 1px solid #24513C; border-radius: 10px; padding: 8px 11px; font-size: 10px; font-weight: 800; }
QFrame#Separator { background: #252A37; border: none; }
QPushButton#NavButton { background: transparent; color: #9BA4B5; border: 1px solid transparent; border-radius: 9px; text-align: left; padding: 12px 13px; font-weight: 600; }
QPushButton#NavButton:hover { background: #1A1F2B; color: #F5F7FB; }
QPushButton#NavButton:checked { background: #252039; color: #D9C9FF; border: 1px solid #403263; }
QFrame#PrivacyCard { background: #171D25; border: 1px solid #2B3442; border-radius: 12px; }
QLabel#PrivacyTitle { color: #83E3B1; font-size: 10px; font-weight: 800; }
QPushButton#SidebarUtility { background: transparent; color: #A9B1C0; border: 1px solid #2B303C; border-radius: 8px; padding: 9px 8px; font-size: 11px; }
QPushButton#SidebarUtility:hover { background: #202532; color: #FFFFFF; }
QFrame#WelcomeCard, QFrame#FeatureHero { background: #131722; border: 1px solid #292F3E; border-radius: 15px; }
QLabel#WelcomeTitle { color: #F7F7FB; font-size: 18px; font-weight: 750; }
QLabel#FeatureTitle { color: #F7F7FB; font-size: 30px; font-weight: 800; padding-top: 3px; padding-bottom: 3px; }
QFrame#FeatureRow { background: #121620; border: 1px solid #272D3A; border-radius: 11px; }
QLabel#FeatureNumber { color: #A78BFA; font-size: 12px; font-weight: 800; }
QLabel#FeatureRowText { color: #E3E7EF; font-size: 13px; font-weight: 600; }
QTextEdit#Transcript { background: #0E1119; color: #E5E7EB; border: 1px solid #242A37; border-radius: 14px; padding: 16px; font-size: 14px; selection-background-color: #6D4BD1; }
QFrame#Composer { background: #141822; border: 1px solid #2C3240; border-radius: 13px; }
QLineEdit#PromptInput, QLineEdit, QComboBox { background: #10141D; color: #F5F7FB; border: 1px solid #303747; border-radius: 8px; padding: 10px 12px; selection-background-color: #6D4BD1; }
QLineEdit#PromptInput { border: 1px solid transparent; background: #141822; font-size: 14px; padding: 10px 4px; }
QLineEdit:focus, QComboBox:focus { border: 1px solid #8B5CF6; }
QComboBox::drop-down { border: none; width: 24px; }
QComboBox QAbstractItemView { background: #151A25; color: #E5E7EB; selection-background-color: #403263; border: 1px solid #303747; }
QPushButton#PrimaryButton { background: #8B5CF6; color: #FFFFFF; border: none; border-radius: 9px; padding: 10px 17px; font-weight: 800; }
QPushButton#PrimaryButton:hover { background: #9B75FA; }
QPushButton#SecondaryButton { background: #202532; color: #D8DDE8; border: 1px solid #343B4B; border-radius: 9px; padding: 9px 12px; font-weight: 650; }
QPushButton#SecondaryButton:hover { background: #2B3140; }
QPushButton:disabled { background: #252A34; color: #687184; }
QScrollBar:vertical { background: #11151E; width: 10px; margin: 0; }
QScrollBar::handle:vertical { background: #343B4B; border-radius: 5px; min-height: 24px; }
QScrollBar::handle:vertical:hover { background: #62518C; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QMessageBox { background: #141822; }
"""
