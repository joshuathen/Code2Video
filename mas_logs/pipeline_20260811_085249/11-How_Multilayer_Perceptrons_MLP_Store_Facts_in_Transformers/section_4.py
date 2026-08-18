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
        self.setup_layout("The Overlapping Fact Storage", [
            "Facts are stored in weight superpositions.",
            "Multiple facts share the same neuron clusters.",
            "Memory is distributed like a hologram."
        ])
        
        # Assets
        hologram = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/hologram.svg")
        neuron = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/neuron.svg")
        
        # Elements
        storage_label = Text("Overlapping Storage", font_size=24, color=WHITE)
        self.place_at_grid(storage_label, 'A3', scale_factor=1.0)
        
        pattern_1 = VGroup(*[Circle(radius=0.15, color=YELLOW, fill_opacity=0.5) for _ in range(5)])
        pattern_1.arrange(RIGHT, buff=0.1)
        self.place_in_area(pattern_1, 'B2', 'B4', scale_factor=0.9)

        pattern_2 = VGroup(*[Square(side_length=0.2, color=YELLOW, fill_opacity=0.5) for _ in range(5)])
        pattern_2.arrange(RIGHT, buff=0.1)
        self.place_in_area(pattern_2, 'D2', 'D4', scale_factor=0.9)

        interference = Circle(radius=0.3, color=PINK, fill_opacity=0.8)
        self.place_at_grid(interference, 'C3', scale_factor=0.8)
        interference.set_opacity(0)
        
        self.place_at_grid(hologram, "E2", scale_factor=0.5)
        self.place_at_grid(neuron, "E5", scale_factor=0.5)
        hologram.set_opacity(0)
        neuron.set_opacity(0)

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(storage_label))
        self.lecture[0].set_color("#FFFFFF")

        # === Animation for Lecture Line 2 ===
        self.play(Create(pattern_1), Create(pattern_2), FadeIn(hologram))
        self.lecture[1].set_color("#FFFF00")

        # === Animation for Lecture Line 3 ===
        self.play(FadeOut(pattern_1), FadeIn(neuron))
        self.lecture[2].set_color("#00FFFF")
        self.play(FadeIn(interference))
        interference.set_color("#FF00FF")
        self.wait(1)
