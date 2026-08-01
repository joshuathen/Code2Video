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

class Section3Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            'Place point A in air and B in water.',
            'Light hits the interface at some point x.',
            'Use Pythagoras to find distance in air.',
            'Repeat this for the distance in water.',
            'Distances are now expressed in terms of x.'
        ]
        self.setup_layout("Setting Up the Geometric Model", lecture_lines)

        # Helper for color updates
        def update_lecture_color(index, color):
            self.play(self.lecture[index].animate.set_color(color), run_time=0.5)

        # Colors
        COLOR_AIR = "#87CEEB"
        COLOR_WATER = "#1E90FF"
        COLOR_RAY = "#FFFF00"
        COLOR_LABEL = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        update_lecture_color(0, YELLOW)
        
        # Interface line
        interface = Line(self.grid["D1"], self.grid["D6"], color=WHITE)
        interface_label = Text("Interface (y=0)", font_size=18).next_to(interface, LEFT, buff=0.1)
        
        # Points A and B
        point_a = Dot(self.grid["B2"], color=COLOR_AIR)
        label_a = Text("A (0, a)", font_size=20).next_to(point_a, UP)
        
        point_b = Dot(self.grid["E5"], color=COLOR_WATER)
        label_b = Text("B (w, -b)", font_size=20).next_to(point_b, DOWN)

        # Coordinate helpers (Vertical lines for a and b)
        dashed_a = DashedLine(self.grid["B2"], self.grid["D2"], color=GRAY)
        label_side_a = Text("a", font_size=20, color=COLOR_LABEL).next_to(dashed_a, LEFT)
        
        dashed_b = DashedLine(self.grid["D5"], self.grid["E5"], color=GRAY)
        label_side_b = Text("b", font_size=20, color=COLOR_LABEL).next_to(dashed_b, RIGHT)
        
        self.play(Create(interface), Write(interface_label))
        self.play(FadeIn(point_a), Write(label_a), Create(dashed_a), Write(label_side_a))
        self.play(FadeIn(point_b), Write(label_b), Create(dashed_b), Write(label_side_b))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        update_lecture_color(1, YELLOW)
        
        point_x = Dot(self.grid["D3"], color=COLOR_RAY)
        label_x = Text("X (x, 0)", font_size=20).next_to(point_x, DOWN)
        
        # Labels for x and w-x segments
        brace_x = BraceBetweenPoints(self.grid["D2"], self.grid["D3"], UP, color=WHITE)
        label_val_x = Text("x", font_size=18).next_to(brace_x, UP, buff=0.1)
        
        brace_w_x = BraceBetweenPoints(self.grid["D3"], self.grid["D5"], UP, color=WHITE)
        label_val_w_x = Text("w - x", font_size=18).next_to(brace_w_x, UP, buff=0.1)
        
        self.play(FadeIn(point_x), Write(label_x))
        self.play(GrowFromCenter(brace_x), Write(label_val_x))
        self.play(GrowFromCenter(brace_w_x), Write(label_val_w_x))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        update_lecture_color(2, YELLOW)
        
        d1_ray = Line(point_a.get_center(), point_x.get_center(), color=COLOR_RAY)
        label_d1 = Text("d₁", font_size=24, color=COLOR_RAY).move_to(
            self.grid["C2"] + RIGHT * 0.5
        )
        
        formula_d1 = Text("d₁ = √(x² + a²)", font_size=24, color=COLOR_RAY)
        # Issue 36: Moving formula_d1 to A5 to avoid clutter near point A
        self.place_at_grid(formula_d1, "A5", scale_factor=0.8)
        
        self.play(Create(d1_ray), Write(label_d1))
        self.play(Write(formula_d1))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        update_lecture_color(3, YELLOW)
        
        d2_ray = Line(point_x.get_center(), point_b.get_center(), color=COLOR_RAY)
        label_d2 = Text("d₂", font_size=24, color=COLOR_RAY).move_to(
            self.grid["E4"] + LEFT * 0.2
        )
        
        formula_d2 = Text("d₂ = √((w - x)² + b²)", font_size=24, color=COLOR_RAY)
        # Issue 35: Moving formula_d2 to B5 to avoid overlap with interface segments
        self.place_at_grid(formula_d2, "B5", scale_factor=0.8)
        
        self.play(Create(d2_ray), Write(label_d2))
        self.play(Write(formula_d2))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        update_lecture_color(4, YELLOW)
        
        # Issue 37: Re-positioning the formula group and highlighting it
        formula_group = VGroup(formula_d1, formula_d2)
        # Smoothly move to the final consolidated area
        target_group = formula_group.copy()
        self.place_in_area(target_group, "A5", "B6", scale_factor=0.7)
        
        self.play(Transform(formula_group, target_group))
        
        box = SurroundingRectangle(formula_group, color=COLOR_RAY, buff=0.2)
        self.play(Create(box))
        self.wait(2)
