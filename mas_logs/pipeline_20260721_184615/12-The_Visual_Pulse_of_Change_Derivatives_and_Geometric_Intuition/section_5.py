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

class Section5Scene(TeachingScene):
    def construct(self):
        # Fetching data from storyboard
        title_text = "Visualizing Special Cases: Constants and Identity"
        lecture_lines = [
            "A flat floor has no tilt or change.",
            "Thus, the derivative of any constant is always zero.",
            "A diagonal line has a constant slope of one."
        ]
        self.setup_layout(title_text, lecture_lines)
        
        # Colors (L008)
        COLOR_CONST = "#ADFF2F"  # Green-yellow
        COLOR_IDENTITY = "#FFA500"  # Orange
        WHITE_COLOR = "#FFFFFF"

        # Axes setup (L001: Grid B2 to F6)
        axes = Axes(
            x_range=[0, 4, 1],
            y_range=[0, 4, 1],
            x_length=4,
            y_length=4,
            axis_config={"include_tip": True, "color": WHITE_COLOR}
        )
        # Place axes so that (0,0) is at grid F2
        self.place_in_area(axes, "B2", "F6")
        
        # === Animation for Lecture Line 1 ===
        # "A flat floor has no tilt or change."
        self.lecture[0].set_color(COLOR_CONST)
        
        # Draw horizontal line y=3
        line_const = axes.plot(lambda x: 3, color=COLOR_CONST, x_range=[0, 4])
        
        # Asset integration (Issue 21)
        # Load floor icon and place it on the line
        floor_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/floor.svg")
        floor_asset.set_color(COLOR_CONST)
        # Positioning asset: Row C is at y=3 relative to axes base F2
        self.place_in_area(floor_asset, "C3", "C5", scale_factor=0.6)
        
        # Labels with VideoCritic fixes (Issue 34)
        label_y3 = MathTex("y = 3", color=COLOR_CONST)
        self.place_at_grid(label_y3, 'A3', scale_factor=0.7)
        
        label_deriv0 = MathTex("y' = 0", color=COLOR_CONST)
        self.place_at_grid(label_deriv0, 'A2', scale_factor=0.7)

        self.play(Create(axes))
        self.play(Create(line_const), DrawBorderThenFill(floor_asset), Write(label_y3))
        self.play(Write(label_deriv0))
        self.wait(2.0)

        # === Animation for Lecture Line 2 ===
        # "Thus, the derivative of any constant is always zero."
        self.lecture[1].set_color(COLOR_CONST)
        
        # Indicator for zero slope (horizontal arrows)
        indicator = DoubleArrow(
            start=axes.c2p(1, 3.2),
            end=axes.c2p(3, 3.2),
            buff=0,
            color=COLOR_CONST,
            tip_length=0.1
        )
        
        slope_text = Text("Zero Slope", font_size=20, color=COLOR_CONST)
        # VideoCritic fix (Issue 34): Area A4-A6
        self.place_in_area(slope_text, 'A4', 'A6', scale_factor=0.7)
        
        self.play(Create(indicator), Write(slope_text))
        self.play(Indicate(indicator)) # L004
        self.wait(2.0)
        
        # Clear indicators for transition
        self.play(FadeOut(indicator), FadeOut(slope_text), FadeOut(floor_asset))

        # === Animation for Lecture Line 3 ===
        # "A diagonal line has a constant slope of one."
        self.lecture[2].set_color(COLOR_IDENTITY)
        
        # Transition line and labels
        line_identity = axes.plot(lambda x: x, color=COLOR_IDENTITY, x_range=[0, 4])
        
        # VideoCritic fixes (Issue 34)
        label_yx = MathTex("y = x", color=COLOR_IDENTITY)
        self.place_at_grid(label_yx, 'A6', scale_factor=0.7)
        
        label_deriv1 = MathTex("y' = 1", color=COLOR_IDENTITY)
        self.place_at_grid(label_deriv1, 'A5', scale_factor=0.7)
        
        # 45-degree angle arc at origin (F2)
        arc = Arc(
            radius=0.7, 
            start_angle=0, 
            angle=PI/4, 
            arc_center=axes.c2p(0,0), 
            color=COLOR_IDENTITY
        )
        angle_label = MathTex("45^\\circ", color=COLOR_IDENTITY)
        # Position label near arc (E3)
        self.place_at_grid(angle_label, "E3", scale_factor=0.6)

        self.play(
            Transform(line_const, line_identity),
            Transform(label_y3, label_yx),
            Transform(label_deriv0, label_deriv1),
            run_time=2
        )
        self.play(Create(arc), Write(angle_label))
        self.wait(2.0)
