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
        lecture_lines = ["Factual information is distributed across layer weights.", 
                         "Think of it like a holographic memory image.", 
                         "Individual neurons contribute to a collective representation.", 
                         "Muting one neuron does not erase the fact.", 
                         "Removing the entire layer deletes the stored information."]
        self.setup_layout("Dynamic Synthesis: From Weights to Facts", lecture_lines)
        
        # Assets
        neuron_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/neuron.svg")
        hologram_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/hologram.svg")
        
        # === Animation for Lecture Line 1 ===
        # Factual information is distributed across layer weights.
        weight_matrix = Matrix([[1, 0.5, -0.2], [0.3, 0.8, 0.1], [-0.5, 0.2, 0.9]], color="#FF5733")
        neuron_icon.set_color("#FF5733")
        combined_matrix_view = VGroup(weight_matrix, neuron_icon).arrange(RIGHT, buff=0.2)
        self.place_in_area(combined_matrix_view, 'A2', 'B3', scale_factor=0.7)
        self.play(FadeIn(combined_matrix_view))
        self.lecture[0].set_color("#FF5733")

        # === Animation for Lecture Line 2 ===
        # Think of it like a holographic memory image.
        hologram_text = Text("Holographic Image", font_size=24, color="#33FF57")
        hologram_icon.set_color("#3357FF")
        combined_hologram_view = VGroup(hologram_text, hologram_icon).arrange(RIGHT, buff=0.2)
        self.place_at_grid(combined_hologram_view, 'C5', scale_factor=0.7)
        self.play(Write(hologram_text), FadeIn(hologram_icon))
        self.lecture[1].set_color("#33FF57")

        # === Animation for Lecture Line 3 ===
        # Individual neurons contribute to a collective representation.
        self.lecture[2].set_color("#3357FF")
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Muting one neuron does not erase the fact.
        self.lecture[3].set_color("#FF33A8")
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Removing the entire layer deletes the stored information.
        self.lecture[4].set_color("#A833FF")
        self.play(FadeOut(combined_matrix_view), FadeOut(combined_hologram_view))
        self.wait(1)
