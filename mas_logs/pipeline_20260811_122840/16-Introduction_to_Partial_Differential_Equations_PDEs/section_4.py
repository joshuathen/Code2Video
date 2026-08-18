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

class Section4Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Elliptic equations describe steady-state systems.",
            "Parabolic equations model diffusion processes.",
            "Hyperbolic equations track wave propagation."
        ]
        self.setup_layout("The Big Three: Classification", lecture_lines)
        
        # --- Create Visual Assets ---
        # 1. Elliptic (Arch)
        arch = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/arch.svg", color="#FFFFFF")
        arch_label = Text("Elliptic", font_size=24, color="#FFFFFF").scale(0.75)
        arch_label.next_to(arch, DOWN)
        elliptic_group = VGroup(arch, arch_label)

        # 2. Parabolic (Cloud)
        cloud = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cloud.svg", color="#FFFF00")
        cloud_label = Text("Parabolic", font_size=24, color="#FFFF00").scale(0.75)
        cloud_label.next_to(cloud, DOWN)
        parabolic_group = VGroup(cloud, cloud_label)

        # 3. Hyperbolic (String)
        string = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/string.svg", color="#FF00FF")
        string_label = Text("Hyperbolic", font_size=24, color="#FF00FF").scale(0.75)
        string_label.next_to(string, DOWN)
        hyperbolic_group = VGroup(string, string_label)

        # === Animation for Lecture Line 1 ===
        # Using B6 for elliptic group as requested
        self.play(self.lecture[0].animate.set_color("#FFFFFF"), FadeIn(self.place_at_grid(elliptic_group, 'B6', scale_factor=0.9)))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFFF00"), FadeIn(self.place_at_grid(parabolic_group, 'D5', scale_factor=0.9)))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Using F6 for hyperbolic group as requested
        self.play(self.lecture[2].animate.set_color("#FF00FF"), FadeIn(self.place_at_grid(hyperbolic_group, 'F6', scale_factor=0.9)))
        self.wait(2)
