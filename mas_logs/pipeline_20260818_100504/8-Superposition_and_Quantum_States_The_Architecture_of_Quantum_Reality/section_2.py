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
        lecture_lines = [
            "Quantum states combine basis states linearly.",
            "Superposition follows: ψ = α|0⟩ + β|1⟩.",
            "α and β are complex probability amplitudes.",
            "Squared amplitudes define the outcome probabilities.",
            "Visualize this as a vector between states."
        ]
        self.setup_layout("Defining Superposition", lecture_lines)
        
        # Elements
        axes = Axes(x_range=[-0.5, 1.5], y_range=[-0.5, 1.5], axis_config={"include_tip": True}).scale(0.5)
        self.place_in_area(axes, "B3", "E5", scale_factor=0.6)
        
        # Vectors using axes coordinates
        vec0 = Arrow(axes.c2p(0, 0), axes.c2p(1, 0), color="#00FF00", buff=0)
        vec1 = Arrow(axes.c2p(0, 0), axes.c2p(0, 1), color="#FF0000", buff=0)
        
        label0 = MathTex(r"|0\rangle", color="#00FF00").next_to(vec0.get_end(), RIGHT)
        label1 = MathTex(r"|1\rangle", color="#FF0000").next_to(vec1.get_end(), UP)
        
        # Icon placeholder for asset
        asset_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg").scale(0.5)
        self.place_at_grid(asset_icon, "A5")
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#00FF00")
        self.play(Create(axes), Create(vec0), Write(label0))
        self.lecture[0].set_color(WHITE)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FF0000")
        self.play(Create(vec1), Write(label1))
        self.lecture[1].set_color(WHITE)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFFF00")
        psi_vec = Arrow(axes.c2p(0, 0), axes.c2p(0.8, 0.6), color="#FFFFFF", buff=0)
        psi_label = MathTex(r"|\psi\rangle = \alpha|0\rangle + \beta|1\rangle", font_size=24)
        self.place_at_grid(psi_label, "B2", scale_factor=0.7)
        self.play(Create(psi_vec), Write(psi_label))
        self.lecture[2].set_color(WHITE)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#00FFFF")
        prob_label = MathTex(r"|\alpha|^2 + |\beta|^2 = 1", font_size=24)
        self.place_at_grid(prob_label, "E2", scale_factor=0.7)
        self.play(Write(prob_label))
        self.lecture[3].set_color(WHITE)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#FF00FF")
        # Rotating vector animation
        angle = ValueTracker(0)
        psi_vec.add_updater(lambda m: m.become(
            Arrow(axes.c2p(0, 0), axes.c2p(np.cos(angle.get_value()), np.sin(angle.get_value())), color="#FFFFFF", buff=0)
        ))
        self.play(angle.animate.set_value(PI/2), run_time=2)
        psi_vec.remove_updater(psi_vec.get_updaters()[0])
        
        self.play(FadeIn(asset_icon))
        self.lecture[4].set_color(WHITE)
        self.wait(2)
