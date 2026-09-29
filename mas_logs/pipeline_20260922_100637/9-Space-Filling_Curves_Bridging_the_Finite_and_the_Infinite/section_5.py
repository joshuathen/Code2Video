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
        lecture_lines = [
            "Space-filling curves optimize modern memory.",
            "They pack antennas into small spaces.",
            "Infinity is a powerful mathematical tool."
        ]
        self.setup_layout("Real-World Application & Closing", lecture_lines)
        
        # --- Create objects using assets ---
        # Note: SVG assets assumed to be loaded via SVGMobject
        antenna = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/antenna.svg")
        antenna.set_color("#00CED1")
        
        waves = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/radio.svg")
        waves.set_color("#FFD700")
        
        label = Text("Space-Filling Antenna", font_size=24, color=WHITE)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(self.place_in_area(antenna, 'B4', 'C6', scale_factor=1.2)))
        self.lecture[0].set_color("#00CED1")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(FadeIn(self.place_in_area(waves, 'E4', 'F6', scale_factor=0.9)))
        self.play(waves.animate.shift(UP*0.5).set_opacity(0.5))
        self.lecture[1].set_color("#FFD700")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(FadeIn(self.place_at_grid(label, 'B5', scale_factor=0.8)))
        self.lecture[2].set_color("#FFFFFF")
        self.wait(2)
