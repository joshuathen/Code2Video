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

class Section3Scene(TeachingScene):
    def construct(self):
        # Initialize standard layout
        lecture_lines = [
            'An ODE relates a function to its derivatives.',
            'We solve for functions, not just single numbers.',
            'This equation locks the value and slope together.',
            'For example, growth increases as the value rises.',
            "The function's behavior depends on its current state."
        ]
        self.setup_layout("Defining the ODE: The Puzzle", lecture_lines)

        # Colors based on prompt
        COLOR_EQ = "#00FFFF"
        COLOR_HIGHLIGHT = "#FFFF00"
        COLOR_SEGMENT = "#00FF00"
        COLOR_FIELD = "#FFFFFF"
        COLOR_CURVE = "#FFA500"

        # === Animation for Lecture Line 1 ===
        # Using Text and VGroup to avoid LaTeX dependency (MathTex)
        self.lecture[0].set_color(COLOR_EQ)
        equation = VGroup(
            Text("dy/dx"), 
            Text("="), 
            Text("y")
        ).arrange(RIGHT, buff=0.2).set_color(COLOR_EQ)
        
        # Position in top half of right side
        # Resolved Issue 40: Changed area and scale to avoid cramping top
        self.place_in_area(equation, "B2", "B5", scale_factor=1.1)
        self.play(Write(equation))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(COLOR_HIGHLIGHT)
        
        y_var = equation[2]
        label_y = Text("unknown function", font_size=18, color=COLOR_HIGHLIGHT)
        label_y.next_to(y_var, DOWN, buff=0.3)
        
        glow_box = SurroundingRectangle(y_var, color=COLOR_HIGHLIGHT, buff=0.1)
        
        self.play(Create(glow_box), Create(label_y))
        # Pulsating effect
        for _ in range(2):
            self.play(y_var.animate.scale(1.2), run_time=0.4)
            self.play(y_var.animate.scale(1/1.2), run_time=0.4)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(COLOR_SEGMENT)
        
        # Transitioning to graph view: Fade equation parts
        self.play(FadeOut(equation), FadeOut(label_y), FadeOut(glow_box))
        
        # Coordinate system on the grid area
        # Resolved Issue 41: Adjusted area and scale to ensure clearance from notes
        axes = Axes(
            x_range=[-2, 2, 1],
            y_range=[0, 4, 1],
            x_length=4.5,
            y_length=3.2,
            axis_config={"include_tip": True, "color": GREY_B}
        )
        self.place_in_area(axes, "C2", "F6", scale_factor=0.85)
        self.play(Create(axes))
        
        # Point (0, 1) has y=1, so slope = 1
        p_scene = axes.c2p(0, 1)
        angle = np.arctan(1)
        length = 0.5
        dx = length * np.cos(angle)
        dy = length * np.sin(angle)
        
        segment = Line(
            p_scene - np.array([dx, dy, 0]),
            p_scene + np.array([dx, dy, 0]),
            color=COLOR_SEGMENT,
            stroke_width=6
        )
        dot = Dot(p_scene, color=COLOR_SEGMENT, radius=0.06)
        seg_label = Text("slope = y", font_size=20, color=COLOR_SEGMENT)
        seg_label.next_to(segment, UR, buff=0.1)
        
        self.play(Create(dot), Create(segment), Write(seg_label))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(COLOR_FIELD)
        
        # Construct slope field manually
        field = VGroup()
        x_pts = np.arange(-1.5, 1.6, 0.75)
        y_pts = np.arange(0.5, 3.6, 0.75)
        for x in x_pts:
            for y in y_pts:
                slope = y
                ang = np.arctan(slope)
                v_len = 0.2
                dx_v = v_len * np.cos(ang)
                dy_v = v_len * np.sin(ang)
                mid = axes.c2p(x, y)
                s = Line(
                    mid - np.array([dx_v, dy_v, 0]),
                    mid + np.array([dx_v, dy_v, 0]),
                    color=COLOR_FIELD,
                    stroke_width=2
                )
                field.add(s)
        
        self.play(FadeOut(seg_label), FadeOut(segment), FadeOut(dot))
        self.play(Create(field))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(COLOR_CURVE)
        
        # Curve y = exp(x)
        curve = axes.plot(lambda x: np.exp(x), x_range=[-2, 1.3], color=COLOR_CURVE)
        self.play(Create(curve))
        self.wait(2)
