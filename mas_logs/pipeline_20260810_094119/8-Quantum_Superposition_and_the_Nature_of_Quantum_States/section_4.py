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
            "Superposition enables massive quantum parallelism.",
            "Process multiple states simultaneously.",
            "Quantum efficiency exceeds classical limits."
        ]
        self.setup_layout("Quantum Parallelism (Application)", lecture_lines)
        
        # Animations
        # === Animation for Lecture Line 1 ===
        # Visualize two parallel computational paths using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/computer.svg] in #00CED1 (DarkTurquoise).
        comp1 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/computer.svg", color="#00CED1")
        comp2 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/computer.svg", color="#00CED1")
        self.place_at_grid(comp1, 'B3', scale_factor=0.8)
        self.place_at_grid(comp2, 'B4', scale_factor=0.8)
        self.play(FadeIn(comp1), FadeIn(comp2))
        self.lecture[0].set_color("#00CED1")

        # === Animation for Lecture Line 2 ===
        # Show both paths being processed simultaneously in #FFFF00 (Yellow).
        dot1 = Dot(color="#FFFF00")
        dot2 = Dot(color="#FFFF00")
        self.place_at_grid(dot1, 'D3', scale_factor=1.0)
        self.place_at_grid(dot2, 'D4', scale_factor=1.0)
        
        path1 = Line(start=comp1.get_bottom(), end=dot1.get_top(), color="#FFFF00")
        path2 = Line(start=comp2.get_bottom(), end=dot2.get_top(), color="#FFFF00")
        
        self.play(Create(path1), Create(path2), FadeIn(dot1), FadeIn(dot2))
        self.lecture[1].set_color("#FFFF00")

        # === Animation for Lecture Line 3 ===
        # Highlight the exponential density growth using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/microchip.svg] in #FF4500 (OrangeRed).
        chip = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/microchip.svg", color="#FF4500")
        self.place_at_grid(chip, 'E4', scale_factor=0.6)
        growth_text = Text("Exponential Density!", font_size=24, color="#FF4500")
        self.place_in_area(growth_text, 'E2', 'F5', scale_factor=0.6)
        self.play(Write(chip), Write(growth_text))
        self.lecture[2].set_color("#FF4500")
        self.wait(2)
