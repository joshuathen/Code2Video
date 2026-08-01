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
        # Setup the scene layout
        self.setup_layout(
            "Prerequisite Knowledge & The Formal Definition", 
            [
                "A circle's width is known as the diameter.",
                "Its boundary length is called the circumference.",
                "Watch the circle roll along a number line.",
                "One full rotation lands precisely at 3.14159.",
                "The ratio of circumference to diameter defines pi."
            ]
        )
        
        # Define constants
        RADIUS = 0.5
        
        # === Animation for Lecture Line 1 ===
        # A circle (#00FFFF) appears with its diameter (#FFFFFF) labeled 'd'.
        
        self.lecture[0].set_color("#00FFFF")
        
        circle = Circle(radius=RADIUS, color="#00FFFF")
        diameter = Line(LEFT * RADIUS, RIGHT * RADIUS, color="#FFFFFF")
        label_d = Text("d", color="#FFFFFF", font_size=30)
        label_d.next_to(diameter, UP, buff=0.1)
        
        circle_group = VGroup(circle, diameter, label_d)
        # Issue 27: Place in area B2 to D6 to avoid boundary tension
        self.place_in_area(circle_group, 'B2', 'D6', scale_factor=0.9)
        
        self.play(Create(circle), Create(diameter), Write(label_d))
        self.wait(1)
        
        # === Animation for Lecture Line 2 ===
        # The circle's circumference glows neon green (#00FF00) and is labeled 'C'.
        
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#00FF00")
        
        label_c = Text("C", color="#00FF00", font_size=30)
        label_c.next_to(circle, UP, buff=0.1)
        
        self.play(
            circle.animate.set_stroke(color="#00FF00", width=6),
            Write(label_c)
        )
        circle_group.add(label_c)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # The circle rolls one full rotation along a horizontal number line.
        
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(WHITE)
        
        number_line = NumberLine(
            x_range=[0, 5, 1],
            length=5,
            include_numbers=True,
            label_constructor=Text,
            font_size=20,
            color=WHITE
        )
        # Issue 28: Position number line along E2-E6
        self.place_in_area(number_line, 'E2', 'E6', scale_factor=1.0)
        
        # Calculate start position based on number line
        start_pos = number_line.n2p(0) + UP * RADIUS
        
        self.play(
            circle_group.animate.move_to(start_pos),
            Create(number_line)
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # One full rotation lands exactly at 3.14159... highlighted with a dot.
        
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color("#FF69B4")
        
        roll_tracker = ValueTracker(0)
        
        def update_circle_roll(m):
            angle = roll_tracker.get_value()
            dist = angle * RADIUS
            m.move_to(start_pos + RIGHT * dist)
            m.set_rotation(-angle)

        circle_group.add_updater(update_circle_roll)
        
        # One full rotation is TAU radians. Distance is TAU * 0.5 = PI.
        self.play(
            roll_tracker.animate.set_value(TAU),
            run_time=4,
            rate_func=linear
        )
        circle_group.remove_updater(update_circle_roll)
        
        # Highlight the landing point 3.14159
        pi_point = number_line.n2p(PI)
        dot = Dot(pi_point, color="#FF69B4")
        pi_label = Text("3.14159...", color="#FF69B4", font_size=24)
        pi_label.next_to(dot, DOWN, buff=0.2)
        
        self.play(FadeIn(dot), Write(pi_label))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # The formula 'pi = C / d' appears in neon pink (#FF69B4).
        
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color("#FF69B4")
        
        # Replacing MathTex with Text to avoid LaTeX dependency error
        formula = Text("π = C / d", color="#FF69B4", font_size=48)
        # Issue 26: Positioning the formula at A3-A6 with scaling
        self.place_in_area(formula, 'A3', 'A6', scale_factor=0.7)
        
        self.play(Write(formula))
        self.wait(3)
