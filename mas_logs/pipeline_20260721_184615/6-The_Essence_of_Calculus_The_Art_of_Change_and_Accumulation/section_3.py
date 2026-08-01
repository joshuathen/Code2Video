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
        # Data from shared state
        title_str = "Differentiation: The Art of Zooming In"
        lecture_lines = [
            "Differentiation finds the exact rate of change right now.",
            "Zoom into any smooth curve far enough.",
            "Eventually, the curve looks like a simple straight line.",
            "This line's slope is the derivative at that point.",
            "It measures instantaneous velocity in a single micro-second."
        ]
        
        self.setup_layout(title_str, lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Draw a cyan #00FFFF curve and mark a specific point P with a white #FFFFFF dot.
        self.lecture[0].set_color("#00FFFF")
        
        p_coords = self.grid["D4"]
        # Use a sine-like curve: y = 0.8 * sin(x - x_p) + y_p
        func = lambda x: 0.8 * np.sin(x - p_coords[0]) + p_coords[1]
        # x_range covers the grid area roughly
        curve = FunctionGraph(func, x_range=[p_coords[0]-2.5, p_coords[0]+2.5], color="#00FFFF")
        curve.set_stroke(width=4)
        
        dot_p = Dot(p_coords, color="#FFFFFF")
        label_p = Text("P", color="#FFFFFF").scale(0.6)
        label_p.next_to(dot_p, UR, buff=0.1)
        
        self.add(curve, dot_p, label_p)
        self.play(Create(curve), FadeIn(dot_p), Write(label_p))
        self.wait(2.0)

        # === Animation for Lecture Line 2 ===
        # Zoom into any smooth curve far enough.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#00FFFF")
        
        # Zooming into the curve. We use scale and preserve stroke width.
        zoom_factor_1 = 4
        self.play(
            curve.animate.scale(zoom_factor_1, about_point=p_coords).set_stroke(width=4),
            label_p.animate.next_to(dot_p, UR, buff=0.1),
            run_time=2.5
        )

        # === Animation for Lecture Line 3 ===
        # Eventually, the curve looks like a simple straight line.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#00FFFF")
        
        zoom_factor_2 = 4 # Total zoom 16x
        self.play(
            curve.animate.scale(zoom_factor_2, about_point=p_coords).set_stroke(width=4),
            label_p.animate.next_to(dot_p, UR, buff=0.1),
            run_time=2.5
        )
        self.wait(2.0)

        # === Animation for Lecture Line 4 ===
        # This line's slope is the derivative at that point.
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color("#FF8000")
        
        # Draw an orange #FF8000 tangent line at point P and label it 'Slope'
        # with icon [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/v.svg]
        tangent_line = TangentLine(curve, alpha=0.5, length=5, color="#FF8000", stroke_width=6)
        
        slope_label = Text("Slope", color="#FF8000")
        # Load asset icon
        v_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/v.svg")
        v_icon.set_color("#FF8000")
        v_icon.set_height(slope_label.get_height())
        
        slope_group = VGroup(slope_label, v_icon).arrange(RIGHT, buff=0.2)
        # Issue 27: Position at B5, scale 0.8
        self.place_at_grid(slope_group, "B5", scale_factor=0.8)
        
        self.play(Create(tangent_line), FadeIn(slope_group))
        self.wait(2.0)

        # === Animation for Lecture Line 5 ===
        # It measures instantaneous velocity in a single micro-second.
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color("#FF8000")
        
        # Highlight the concept
        self.play(
            Indicate(slope_group),
            Indicate(dot_p),
            run_time=2.0
        )
        
        # Subtle pulse of the tangent line to emphasize the 'velocity'
        self.play(
            tangent_line.animate.set_stroke(width=10),
            rate_func=rate_functions.there_and_back,
            run_time=2.0
        )
        self.wait(2.0)
