#include <QApplication>
#include "MainWindow.h"

// 程序入口：Qt 的 GUI 程序固定套路
//   1. 造 QApplication（管理整个程序的事件循环、字体、剪贴板等全局资源）
//   2. 造主窗口并 show()
//   3. exec() 进入事件循环——程序从此在这里“活着”，直到窗口关闭 exec 才返回
int main(int argc, char *argv[]) {
    QApplication app(argc, argv);

    MainWindow w;
    w.show();

    return QApplication::exec();
}
