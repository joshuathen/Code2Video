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

class Section5Scene(TeachingScene):
    def construct(self):
        lecture_lines = ["Anti-commutative property: a x b.", "Calculates physical torque force.", "Computes 3D surface normals."]
        self.setup_layout("Summary and Application", lecture_lines)
        
        # Assets
        wrench = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/wrench.svg")
        surface = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/surface.svg")
        bolt = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bolt.svg")

        # 1. Summarize 3D cross product formula
        formula = MathTex(r"\mathbf{a} \times \mathbf{b} = -(\mathbf{b} \times \mathbf{a})", color=WHITE)
        self.place_in_area(formula, 'B2', 'B5', scale_factor=1.0)
        self.place_at_grid(wrench, 'B6', scale_factor=0.5)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(formula), FadeIn(wrench), self.lecture[0].animate.set_color("#FFFFFF"))
        self.wait(1)

        # 2. Display area calculation application
        torque_label = Text("Torque: " + r"$\vec{\tau} = \vec{r} \times \vec{F}$", font_size=24, color="#FF8833")
        self.place_at_grid(torque_label, 'D2', scale_factor=0.9)
        self.place_at_grid(surface, 'D5', scale_factor=0.5)
        
        # === Animation for Lecture Line 2 ===
        self.play(FadeIn(torque_label), FadeIn(surface), self.lecture[1].animate.set_color("#FF8833"))
        self.wait(1)

        # 3. Highlight cross product usage in vector fields
        normal_label = Text("Surface Normal: n = a x b", font_size=24, color="#33FF57")
        self.place_at_grid(normal_label, 'E2', scale_factor=0.9)
        self.place_at_grid(bolt, 'E5', scale_factor=0.5)

        # === Animation for Lecture Line 3 ===
        self.play(FadeIn(normal_label), FadeIn(bolt), self.lecture[2].animate.set_color("#33FF57"))
        self.wait(2)
