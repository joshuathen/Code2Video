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
        lecture_lines = ["Text embeddings guide the denoising.", "They act as a semantic magnet.", "Pixels align with requested descriptions."]
        self.setup_layout("Guiding the Process: The Mathematics of Condition", lecture_lines)
        
        # --- Assets ---
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/pixels.svg]
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/magnet.svg]
        
        pixels_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/pixels.svg", color="#444444")
        magnet_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/magnet.svg", color="#00BFFF")
        
        # --- Visualization elements ---
        denoising_cluster = VGroup(pixels_icon, *[Dot(np.random.randn(3)*0.5, color="#444444") for _ in range(20)])
        prompt_vec = Vector(RIGHT * 1.5, color="#FF69B4")
        prompt_vector_label = Text("Prompt Vector", font_size=18, color="#FF69B4")
        grid_visual_group = VGroup(magnet_icon, prompt_vec)

        # === Animation for Lecture Line 1 ===
        self.place_at_grid(denoising_cluster, 'B4', scale_factor=0.6)
        self.play(FadeIn(denoising_cluster), Create(prompt_vec), Write(prompt_vector_label))
        self.lecture[0].set_color("#444444")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.place_at_grid(prompt_vector_label, 'A4', scale_factor=0.7)
        influence_arrow = CurvedArrow(self.grid["A4"], self.grid["B4"], color="#00BFFF", angle=-TAU/8)
        self.play(Create(influence_arrow), Create(magnet_icon.move_to(self.grid["A6"])))
        self.lecture[1].set_color("#00BFFF")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.place_in_area(grid_visual_group, 'C2', 'F5', scale_factor=0.8)
        self.play(denoising_cluster.animate.set_color("#FFD700"), run_time=2)
        self.lecture[2].set_color("#FFD700")
        self.wait(2)
