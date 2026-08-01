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
        # Data from storyboard
        title = "Prerequisite Knowledge: Pivots and Rotation"
        lines = [
            "A pivot point is the center of rotation.",
            "Rotating a line 180 degrees swaps its sides.",
            "This divides the plane into two distinct halves."
        ]
        self.setup_layout(title, lines)
        
        # Colors
        COLOR_PIVOT = "#FFFF00"
        COLOR_LINE = "#FFFFFF"
        COLOR_REGION_A = "#FF00FF" # Light Magenta
        COLOR_REGION_B = "#00FF00" # Light Green
        
        # === Animation for Lecture Line 1 ===
        # L1: "A pivot point is the center of rotation."
        self.play(self.lecture[0].animate.set_color(COLOR_PIVOT))
        
        # Issue 20: [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/star.svg]
        pivot = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/star.svg")
        pivot.set_color(COLOR_PIVOT)
        self.place_at_grid(pivot, "C3", scale_factor=0.2)
        
        # Issue 25: Multi-word label 'pivot_label' ('Pivot Point')
        pivot_label = Text("Pivot Point", font_size=20, color=COLOR_PIVOT)
        self.place_in_area(pivot_label, 'B2', 'B4', scale_factor=0.8)
        
        # White line passing through the pivot
        rotating_line = Line(LEFT * 3.0, RIGHT * 3.0, color=COLOR_LINE, stroke_width=4)
        rotating_line.move_to(pivot.get_center())
        
        self.play(
            FadeIn(pivot),
            FadeIn(pivot_label),
            Create(rotating_line)
        )
        self.wait(1)
        
        # === Animation for Lecture Line 2 ===
        # L2: "Rotating a line 180 degrees swaps its sides."
        self.play(self.lecture[1].animate.set_color(COLOR_REGION_A))
        
        # Create regions (half-planes)
        region_a = Rectangle(width=6, height=3, color=COLOR_REGION_A, fill_opacity=0.2, stroke_width=0)
        region_b = Rectangle(width=6, height=3, color=COLOR_REGION_B, fill_opacity=0.2, stroke_width=0)
        
        # Position them relative to the line
        region_a.next_to(rotating_line, UP, buff=0)
        region_b.next_to(rotating_line, DOWN, buff=0)
        
        # Issue 26 & 27: Labels for the halves using place_in_area
        label_a = Text("Half-plane 1", font_size=18, color=COLOR_REGION_A)
        label_b = Text("Half-plane 2", font_size=18, color=COLOR_REGION_B)
        self.place_in_area(label_a, 'A4', 'B5', scale_factor=0.7)
        self.place_in_area(label_b, 'E4', 'F5', scale_factor=0.7)
        
        self.play(
            FadeIn(region_a),
            FadeIn(region_b),
            FadeIn(label_a),
            FadeIn(label_b)
        )
        self.wait(1)
        
        # === Animation for Lecture Line 3 ===
        # L3: "This divides the plane into two distinct halves."
        self.play(self.lecture[2].animate.set_color(COLOR_REGION_B))
        
        # Group objects that rotate together
        rotation_group = VGroup(rotating_line, region_a, region_b, label_a, label_b)
        
        self.play(
            Rotate(
                rotation_group,
                angle=PI,
                about_point=pivot.get_center(),
                rate_func=smooth
            ),
            run_time=2.5
        )
        
        # Highlight the final state
        # L004: Use Indicate
        self.play(Indicate(region_a), Indicate(region_b))
        self.wait(2)
