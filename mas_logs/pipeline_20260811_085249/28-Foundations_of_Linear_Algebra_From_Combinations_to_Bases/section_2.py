from manim import *

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
        self.setup_layout("Linear Combinations and Span", [
            "Linear combination: scale and add vectors.",
            "Span: all points reachable by combinations.",
            "Example: robot arm's reachable 2D area."
        ])
        
        # Asset Loading
        robot = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg")
        self.place_at_grid(robot, 'A5', scale_factor=0.5)
        
        # Define vectors
        v = Vector([1, 2], color="#FF00FF")
        w = Vector([2, -1], color="#00FFFF")
        
        # Apply positioning constraints (Moved to columns 4-6)
        self.place_at_grid(v, 'C4', scale_factor=0.6)
        self.place_at_grid(w, 'C6', scale_factor=0.6)
        
        self.add(robot)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FF00FF"))
        self.play(Create(v), Create(w))
        a = ValueTracker(1.5)
        
        # Use simple scaling rather than always_redraw if possible for performance
        # For this, just showing the vectors initially
        self.play(a.animate.set_value(0.5), run_time=1)
        self.play(a.animate.set_value(2.0), run_time=1)
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00FFFF"))
        # Parallelogram representing span, positioned in grid D-F
        span_area = Polygon(
            [0, 0, 0], [1, 2, 0], [3, 1, 0], [2, -1, 0],
            fill_opacity=0.3, color=YELLOW, stroke_width=0
        )
        self.place_in_area(span_area, 'D4', 'F6', scale_factor=0.8)
        self.play(FadeIn(span_area))
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFFF00"))
        self.play(Flash(span_area, color=WHITE, line_length=0.2, num_lines=15))
        self.play(robot.animate.shift(UP * 0.5)) # Celebration
        self.wait(1)
