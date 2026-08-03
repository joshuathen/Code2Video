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
        title = "Visualizing the Math: The Bloch Sphere"
        lines = [
            "We represent quantum states as vectors within a sphere.",
            "The Bloch sphere maps pure states to its surface.",
            "Superposition points anywhere between the poles.",
            "The equation alpha zero plus beta one describes this.",
            "Coefficients represent the weight of each base state."
        ]
        self.setup_layout(title, lines)

        # === Animation for Lecture Line 1 ===
        # Draw a white wireframe sphere [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg] (#FFFFFF)
        self.bloch_sphere = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg").set_color(WHITE)
        # Issue 24: Fix overlap by changing area to 'B2' to 'D5'
        self.place_in_area(self.bloch_sphere, "B2", "D5", scale_factor=1.5)
        
        self.play(
            self.lecture[0].animate.set_color(YELLOW),
            Create(self.bloch_sphere),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Label the North Pole '|0⟩' (#00FFFF) and the South Pole '|1⟩' (#FFA500)
        center = self.bloch_sphere.get_center()
        # Determine radius based on height to position labels precisely at poles
        radius = self.bloch_sphere.height / 2
        north_pole_pos = center + UP * radius
        south_pole_pos = center + DOWN * radius
        
        label_0 = MathTex(r"|0\rangle", color="#00FFFF").scale(0.8)
        label_1 = MathTex(r"|1\rangle", color="#FFA500").scale(0.8)
        
        # Position labels at the poles
        label_0.next_to(north_pole_pos, UP, buff=0.1)
        label_1.next_to(south_pole_pos, DOWN, buff=0.1)
        
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW),
            Write(label_0),
            Write(label_1),
            run_time=1.2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Create a gold vector arrow (#FFD700) starting at the North Pole.
        arrow = Arrow(start=center, end=north_pole_pos, buff=0, color="#FFD700", stroke_width=4)
        
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW),
            Create(arrow),
            run_time=1.0
        )
        self.wait(0.5)
        
        # Rotate the arrow down to the sphere's equator.
        self.play(
            Rotate(arrow, angle=-PI/2, about_point=center),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Display the equation |ψ⟩ = α|0⟩ + β|1⟩ (#FFFFFF)
        self.equation = MathTex(
            r"|\psi\rangle", "=", r"\alpha", r"|0\rangle", "+", r"\beta", r"|1\rangle",
            color=WHITE
        )
        # Issue 25: Fix cramped equation by placing in area 'F1' to 'F6' with scale_factor=0.7
        self.place_in_area(self.equation, 'F1', 'F6', scale_factor=0.7)
        
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(YELLOW),
            Write(self.equation),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Coefficients represent the weight of each base state.
        # equation components: |\psi\rangle(0), =(1), \alpha(2), |0\rangle(3), +(4), \beta(5), |1\rangle(6)
        alpha_rect = SurroundingRectangle(self.equation[2], color=YELLOW, buff=0.1)
        beta_rect = SurroundingRectangle(self.equation[5], color=YELLOW, buff=0.1)
        
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(YELLOW),
            Create(alpha_rect),
            Create(beta_rect),
            run_time=1.0
        )
        self.wait(2)
        
        # Cleanup for end of scene
        self.play(
            FadeOut(alpha_rect),
            FadeOut(beta_rect),
            self.lecture[4].animate.set_color(WHITE),
            run_time=1.0
        )
        self.wait(2)
