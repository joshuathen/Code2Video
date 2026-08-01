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

class Section4Scene(TeachingScene):
    def construct(self):
        # Setup the layout with the title and lecture lines
        lecture_lines = [
            "Logarithmic scales help map energy across various wave numbers.",
            "Scattered data points align into a strict linear trend.",
            "This line reveals Kolmogorov's universal five-thirds scaling law."
        ]
        self.setup_layout("Kolmogorov\u2019s -5/3 Scaling Law", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Draw white #FFFFFF coordinate axes on a logarithmic scale with labels 'E(k)' and 'k'
        self.play(self.lecture[0].animate.set_color(WHITE))
        
        axes = Axes(
            x_range=[0, 6, 1],
            y_range=[0, 6, 1],
            x_length=4.5,
            y_length=4.5,
            axis_config={"color": WHITE, "include_tip": True},
            tips=True
        )
        
        y_label = Text("E(k)", font_size=18, color=WHITE)
        x_label = Text("k", font_size=18, color=WHITE)
        
        # Position axes in the A1-F6 area
        self.place_in_area(axes, "A1", "F6")
        
        # Labels relative to grid-placed axes
        y_label.next_to(axes.y_axis.get_top(), LEFT, buff=0.1)
        x_label.next_to(axes.x_axis.get_right(), DOWN, buff=0.1)
        
        self.play(Create(axes), Write(y_label), Write(x_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Plot a cloud of blue #0000FF dots that instantly snap into a linear trend line.
        self.play(self.lecture[1].animate.set_color(BLUE))
        
        # Generate dots with noise around visual slope: y = -5/3 * x + 8.5
        np.random.seed(42)
        dots = VGroup()
        target_positions = []
        
        for i in range(25):
            x_val = np.random.uniform(1.5, 5)
            y_ideal = - (5/3) * x_val + 8.5
            
            # Scattered initial state
            noise_x = np.random.uniform(-0.6, 0.6)
            noise_y = np.random.uniform(-0.6, 0.6)
            initial_pos = axes.c2p(x_val + noise_x, y_ideal + noise_y)
            
            dot = Dot(point=initial_pos, radius=0.06, color="#0000FF")
            dots.add(dot)
            target_positions.append(axes.c2p(x_val, y_ideal))
            
        self.play(FadeIn(dots, shift=UP))
        self.wait(0.5)

        # Trend line
        line_start = axes.c2p(1.2, - (5/3) * 1.2 + 8.5)
        line_end = axes.c2p(5.0, - (5/3) * 5.0 + 8.5)
        trend_line = Line(line_start, line_end, color="#0000FF", stroke_width=4)
        
        snap_animations = [dot.animate.move_to(target_positions[i]) for i, dot in enumerate(dots)]
        self.play(*snap_animations, run_time=1)
        self.play(Create(trend_line))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Draw a slope indicator triangle next to the line with the text label '-5/3' in orange #FFA500.
        self.play(self.lecture[2].animate.set_color("#FFA500"))
        
        # Create slope triangle
        tri_x1, tri_y1 = 2.0, - (5/3) * 2.0 + 8.5
        tri_x2 = 2.8
        tri_y2 = - (5/3) * 2.8 + 8.5
        
        p1 = axes.c2p(tri_x1, tri_y1)
        p2 = axes.c2p(tri_x2, tri_y1) # corner
        p3 = axes.c2p(tri_x2, tri_y2)
        
        slope_triangle = Polygon(p1, p2, p3, color="#FFA500", stroke_width=3, fill_opacity=0.2)
        slope_label = Text("-5/3", font_size=20, color="#FFA500")
        slope_label.next_to(p2, RIGHT, buff=0.1)
        
        # Formula - Resolved Issue 34: Move to B5 and scale 0.8
        formula = Text("E(k) = C \u03b5\u00b2\u002f\u00b3 k\u207b\u2075\u002f\u00b3", font_size=24, color=WHITE)
        self.place_at_grid(formula, "B5", scale_factor=0.8)
        
        self.play(Create(slope_triangle))
        self.play(Write(slope_label))
        self.play(FadeIn(formula, shift=UP))
        self.play(Indicate(formula, color=WHITE))
        self.wait(2)
