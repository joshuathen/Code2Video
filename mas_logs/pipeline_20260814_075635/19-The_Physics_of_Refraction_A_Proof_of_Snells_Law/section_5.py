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
        lecture_lines = ["Refraction follows the path of least time.", "The lifeguard analogy holds true for light.", "Nature optimizes to find efficiency."]
        self.setup_layout("Conclusion: Real-world Synthesis", lecture_lines)
        
        colors = [BLUE, GREEN, YELLOW]
        lifeguard_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/lifeguard.svg")

        # 1. Recap the lifeguard's path as a light ray.
        # === Animation for Lecture Line 1 ===
        path = Line(self.grid["A2"], self.grid["C4"], color=BLUE).add_tip()
        icon1 = lifeguard_icon.copy()
        self.place_at_grid(icon1, "A1", scale_factor=0.3)
        ray_path_group = VGroup(path, Text("Light ray path", font_size=20).next_to(path, UP, buff=0.1))
        self.place_at_grid(ray_path_group, "A4", scale_factor=0.9)
        self.play(Create(path), Write(ray_path_group[1]), FadeIn(icon1))
        self.lecture[0].set_color(colors[0])
        self.wait(1)

        # 2. Connect the geometric result to physical optics.
        # === Animation for Lecture Line 2 ===
        rect = Rectangle(width=2, height=1, color=GREEN).shift(self.grid["D4"])
        glass_label = Text("Glass medium", font_size=20)
        self.place_at_grid(glass_label, "D3", scale_factor=0.7)
        self.play(Create(rect), Write(glass_label))
        self.lecture[1].set_color(colors[1])
        self.wait(1)

        # 3. Conclude by showing Snell's Law governing nature.
        # === Animation for Lecture Line 3 ===
        snell = MathTex(r"n_1 \sin \theta_1 = n_2 \sin \theta_2", color=YELLOW)
        self.place_in_area(snell, "B2", "C5", scale_factor=1.0)
        icon2 = lifeguard_icon.copy()
        self.place_at_grid(icon2, "E4", scale_factor=0.3)
        self.play(Write(snell), FadeIn(icon2))
        self.lecture[2].set_color(colors[2])
        self.wait(2)
