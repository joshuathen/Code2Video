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

class Section1Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Prerequisite: The Binary vs. The Wave", [
            "Classical bits are either zero or one.",
            "Quantum systems exist as probability waves.",
            "Measurement collapses the wave into state."
        ])
        
        # Define objects
        # Note: Asset path /scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg exists
        bit = Square(side_length=1, color="#FFD700", fill_opacity=0.5)
        bit_label = Text("0", color=WHITE)
        bit_group = VGroup(bit, bit_label)
        
        wave = FunctionGraph(lambda x: 0.5 * np.sin(3 * x), x_range=[-1.5, 1.5], color="#00FFFF")
        wave_label = Text("Ψ(x)", color="#00FFFF", font_size=20)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(self.lecture[0]))
        self.lecture[0].set_color("#FFD700")
        # Fix: Move from C3 to C2 and scale to avoid occlusion
        self.place_at_grid(bit_group, 'C2', scale_factor=0.8)
        self.play(Create(bit_group))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(FadeIn(self.lecture[1]))
        self.lecture[1].set_color("#00FFFF")
        # Fix: Move wave to C4 and scale
        self.place_at_grid(wave, 'C4', scale_factor=0.8)
        # Fix: Move wave_label relative to wave
        self.place_at_grid(wave_label, 'B4', scale_factor=0.7)
        self.play(Create(wave), Write(wave_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(FadeIn(self.lecture[2]))
        self.lecture[2].set_color("#FFFFFF")
        # Visualizing collapse: fade the wave into the bit
        self.play(FadeOut(wave), FadeOut(wave_label), bit.animate.set_color("#FFFFFF"))
        self.wait(2)
