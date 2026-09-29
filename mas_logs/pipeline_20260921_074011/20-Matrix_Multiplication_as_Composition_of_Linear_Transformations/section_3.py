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
        self.setup_layout("Visualizing the 'Combined' Machine", [
            "Can we combine two transformations into one?",
            "Yes, C represents the single combined transformation.",
            "So, matrix multiplication AB results in matrix C.",
            "Think of this as a dual-lens filter.",
            "The two steps become one single operation."
        ])
        
        # Load asset
        machine_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/machine.svg")
        
        # Create elements
        m1 = machine_asset.copy().set_color(WHITE)
        m2 = machine_asset.copy().set_color(WHITE)
        label1 = Text("M1", font_size=24)
        label2 = Text("M2", font_size=24)
        m1_group = VGroup(m1, label1)
        m2_group = VGroup(m2, label2)
        
        m3 = machine_asset.copy().set_color(WHITE)
        label3 = Text("M3 (Combined)", font_size=24)
        m3_group = VGroup(m3, label3)

        data = Circle(radius=0.3, color=YELLOW).set_fill(YELLOW, opacity=0.8)
        
        # === Animation for Lecture Line 1 ===
        # Addressing Issue 25: Reposition m1/m2
        # Addressing Issue 27: Reposition labels
        self.place_at_grid(m1, 'B1', scale_factor=0.6)
        self.place_at_grid(m2, 'B3', scale_factor=0.6)
        self.place_at_grid(label1, 'A1', scale_factor=0.5)
        self.place_at_grid(label2, 'A3', scale_factor=0.5)
        self.play(FadeIn(m1_group), FadeIn(m2_group))
        self.lecture[0].set_color("#FFFFFF")

        # === Animation for Lecture Line 2 ===
        self.play(FadeIn(data))
        data.move_to(self.grid['B1'])
        self.play(data.animate.move_to(self.grid['B3']))
        self.lecture[1].set_color("#FFCC00")

        # === Animation for Lecture Line 3 ===
        output = Star(n=5, color=TEAL).scale(0.3).set_fill(TEAL, opacity=0.8)
        self.play(data.animate.move_to(self.grid['E4']), FadeIn(output))
        self.lecture[2].set_color("#00FFCC")

        # === Animation for Lecture Line 4 ===
        # Addressing Issue 26: Reposition M3
        self.play(FadeOut(m1_group), FadeOut(m2_group), FadeOut(data), FadeOut(output))
        self.place_at_grid(m3, 'C2', scale_factor=0.8)
        self.place_at_grid(label3, 'B2', scale_factor=0.6)
        self.play(FadeIn(m3_group))
        self.lecture[3].set_color("#FF66FF")

        # === Animation for Lecture Line 5 ===
        flow = DashedLine(self.grid['C1'], self.grid['C3'], color=WHITE)
        self.play(Create(flow))
        self.lecture[4].set_color("#FFFFFF")
        self.wait(1)
