from manim import *
import numpy as np

class TeachingScene(Scene):
    def setup_layout(self, title_text, lecture_lines):
        # BASE
        self.camera.background_color = "#000000"
        self.title = Text(title_text, font_size=28, color=WHITE).to_edge(UP)
        self.add(self.title)

        # Left-side lecture content (bullets with "-")
        lecture_texts = [Text(line, font_size=22, color=WHITE) for line in lecture_lines]
        self.lecture = VGroup(*lecture_texts).arrange(DOWN, aligned_edge=LEFT).scale(0.8)
        self.lecture.to_edge(LEFT, buff=0.2)
        self.add(self.lecture)

        # Define fine-grained animation grid (4x4 grid on right side)
        self.grid = {}
        rows = ["A", "B", "C", "D", "E", "F"]  # Top to bottom
        cols = ["1", "2", "3", "4", "5", "6"]  # Left to right

        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                x = 0.5 + j * 1
                y = 2.2 - i * 1
                self.grid[f"{row}{col}"] = np.array([x, y, 0])

    def place_at_grid(self, mobject, grid_pos, scale_factor=1.0):
        mobject.scale(scale_factor)
        mobject.move_to(self.grid[grid_pos])
        return mobject

    def place_in_area(self, mobject, top_left, bottom_right, scale_factor=1.0):
        tl_pos = self.grid[top_left]
        br_pos = self.grid[bottom_right]
        
        # Calculate center of the area
        center_x = (tl_pos[0] + br_pos[0]) / 2
        center_y = (tl_pos[1] + br_pos[1]) / 2
        center = np.array([center_x, center_y, 0])
        
        mobject.scale(scale_factor)
        mobject.move_to(center)
        return mobject

class Section2Scene(TeachingScene):
    def construct(self):
        # Data from storyboard and outline
        title = "Prerequisite Review: Slope and Area"
        lines = [
            "Derivatives measure the slope at a point.",
            "Integrals calculate the area under a curve.",
            "These look like very different mathematical tools."
        ]
        self.setup_layout(title, lines)

        # Colors
        COLOR_SLOPE = "#FF00FF" # Magenta
        COLOR_AREA = "#00FF00"  # Green
        COLOR_HIGHLIGHT = "#FFFF00" # Yellow for active lecture line

        # --- Slope Components ---
        # Issue 33: Adjusted placement to B2-C5 to avoid crowding
        axes_slope = Axes(
            x_range=[0, 4],
            y_range=[0, 20],
            x_length=4,
            y_length=2,
            axis_config={"include_tip": False, "color": GREY_C},
            tips=False
        )
        self.place_in_area(axes_slope, "B2", "C5")
        
        graph_slope = axes_slope.plot(lambda x: 5*x, x_range=[0, 3.5], color=WHITE)
        label_slope_func = Text("x(t) = 5t", color=WHITE, font_size=20)
        self.place_at_grid(label_slope_func, "A5", scale_factor=0.8)
        
        # Slope triangle at t=2 to t=3
        p1 = axes_slope.c2p(2, 10)
        p2 = axes_slope.c2p(3, 10)
        p3 = axes_slope.c2p(3, 15)
        slope_triangle = Polygon(p1, p2, p3, color=COLOR_SLOPE, fill_opacity=0.6, stroke_width=2)
        slope_label = Text("Slope = 5", color=COLOR_SLOPE, font_size=16)
        slope_label.next_to(slope_triangle, RIGHT, buff=0.1)

        # --- Area Components ---
        # Issue 34: Adjusted placement to E2-F5 to improve spacing
        axes_area = Axes(
            x_range=[0, 4],
            y_range=[0, 10],
            x_length=4,
            y_length=2,
            axis_config={"include_tip": False, "color": GREY_C},
            tips=False
        )
        self.place_in_area(axes_area, "E2", "F5")
        
        graph_area = axes_area.plot(lambda x: 5, x_range=[0, 3.5], color=WHITE)
        # Issue 35: Applied scale_factor=0.8 to label_area_func at D5
        label_area_func = Text("v(t) = 5", color=WHITE, font_size=20)
        self.place_at_grid(label_area_func, "D5", scale_factor=0.8)
        
        # Area rectangle from t=0 to t=3
        rect_area = axes_area.get_area(graph_area, x_range=[0, 3], color=COLOR_AREA, opacity=0.4)
        area_label = Text("Area = 15", color=COLOR_AREA, font_size=16)
        area_label.move_to(rect_area.get_center())

        # === Animation for Lecture Line 1 ===
        # "Derivatives measure the slope at a point."
        self.play(self.lecture[0].animate.set_color(COLOR_HIGHLIGHT))
        self.play(
            Create(axes_slope),
            Create(graph_slope),
            Write(label_slope_func)
        )
        self.play(DrawBorderThenFill(slope_triangle), Write(slope_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "Integrals calculate the area under a curve."
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(COLOR_HIGHLIGHT)
        )
        self.play(
            Create(axes_area),
            Create(graph_area),
            Write(label_area_func)
        )
        self.play(FadeIn(rect_area), Write(area_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "These look like very different mathematical tools."
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(COLOR_HIGHLIGHT)
        )
        
        # Pulse both to show connection
        self.play(
            slope_triangle.animate.scale(1.2),
            rect_area.animate.scale(1.2),
            rate_func=there_and_back,
            run_time=2
        )
        self.wait(2)
