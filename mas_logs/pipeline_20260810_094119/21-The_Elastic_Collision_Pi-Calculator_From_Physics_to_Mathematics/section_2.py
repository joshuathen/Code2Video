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
        lecture_lines = ["Energy and momentum must be conserved.", "Elastic collisions preserve total kinetic energy.", "Physics maps to a rotation matrix."]
        self.setup_layout("Prerequisite Physics: Conservation Laws", lecture_lines)
        
        # Mobjects
        eq_momentum = MathTex(r"m_1v_1 + m_2v_2 = \text{const}", font_size=36, color="#FFCC00")
        eq_energy = MathTex(r"\frac{1}{2}m_1v_1^2 + \frac{1}{2}m_2v_2^2 = \text{const}", font_size=36, color="#FFCC00")
        equations = VGroup(eq_momentum, eq_energy).arrange(DOWN, buff=0.5)
        
        axes = Axes(x_range=[-2, 2, 1], y_range=[-2, 2, 1], axis_config={"include_tip": True}).scale(0.5)
        particles = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/particles.svg")
        phase_plot = VGroup(axes, particles)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFCC00"))
        # Fix issue 24/39: place equations at B2-B5 with 0.6 scale
        self.place_in_area(equations, "B2", "B5", scale_factor=0.6)
        self.play(FadeIn(equations))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFCC00"))
        # Keep equations visible, implicitly handled

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFCC00"))
        # Fix issue 25/40: place phase_plot at C2-D5 with 0.5 scale
        self.place_in_area(phase_plot, "C2", "D5", scale_factor=0.5)
        self.play(Create(phase_plot))
        
        matrix = MathTex(r"R(\theta) = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}", font_size=30, color="#FFFFFF")
        # Fix issue 26/41: place matrix at E3 with 0.7 scale
        self.place_at_grid(matrix, "E3", scale_factor=0.7)
        self.play(Write(matrix))
        
        self.wait(2)
