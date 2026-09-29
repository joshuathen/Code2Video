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
            "MLPs contain a Key layer for patterns.",
            "The Value layer stores factual vectors.",
            "Pattern triggers retrieve associated facts.",
            "This mechanism enables rapid fact recall.",
            "Transformers integrate these memories efficiently."
        ]
        self.setup_layout("Key-Value Memories: The MLP Mechanism", lecture_lines)
        
        # --- Visual Objects ---
        # Assets
        key_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/key.svg")
        library_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/library.svg")
        magnet_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/magnet.svg")
        filing_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/filing.svg")
        
        # Other objects
        input_dot = Dot(color="#FF6600")
        weight_mat = Matrix([[1, 0], [0, 1]], v_buff=0.4, h_buff=0.4).set_color(WHITE)
        mlp_layer = Rectangle(width=2, height=1, color=WHITE)
        key_neurons = VGroup(*[Dot(color="#FFFF00") for _ in range(4)]).arrange(DOWN)
        output_vec = Arrow(start=LEFT, end=RIGHT, color="#FF00FF").scale(0.5)

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(self.place_at_grid(input_dot, 'B2')), FadeIn(self.place_at_grid(key_icon, 'B3')))
        self.lecture[0].set_color("#FF6600")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(FadeIn(self.place_in_area(weight_mat, 'C3', 'D4', scale_factor=0.6)), FadeIn(self.place_at_grid(library_icon, 'C5')))
        self.lecture[1].set_color("#00FFCC")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(FadeIn(self.place_in_area(mlp_layer, 'C5', 'D6', scale_factor=0.7)), FadeIn(self.place_at_grid(magnet_icon, 'D2')))
        self.lecture[2].set_color("#FFFFFF")
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(FadeIn(self.place_at_grid(key_neurons, 'E2')))
        key_neurons.set_color("#FFFF00") # Fire animation placeholder
        self.lecture[3].set_color("#FFFF00")
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(FadeIn(self.place_at_grid(output_vec, 'E6', scale_factor=0.7)), FadeIn(self.place_at_grid(filing_icon, 'F6')))
        self.lecture[4].set_color("#FF00FF")
        self.wait(1)
