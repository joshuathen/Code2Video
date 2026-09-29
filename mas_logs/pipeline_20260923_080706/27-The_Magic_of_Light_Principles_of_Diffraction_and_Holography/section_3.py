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
            "Photography only captures light intensity.", 
            "Holography records the wave phase.", 
            "It uses reference and object beams.", 
            "They interfere to create depth.", 
            "This reveals true three-dimensional images."
        ]
        self.setup_layout("Holography: Encoding the Phase", lecture_lines)
        
        # Assets
        laser_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/laser.svg")
        plate_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/plate.svg")
        hologram_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/hologram.svg")

        # Prep elements
        ref_beam = Line(start=self.grid['A3'], end=self.grid['E3'], color="#FF0000")
        obj_beam = Line(start=self.grid['A4'], end=self.grid['E4'], color="#00FF00")
        light_beams = VGroup(laser_icon, ref_beam, obj_beam)
        
        plate = plate_icon.copy()
        interference_label = Text("Interference", font_size=20, color="#FFFFFF")
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFD700")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#00FFFF")
        self.place_in_area(light_beams, 'A3', 'E4', scale_factor=0.7)
        self.play(FadeIn(light_beams))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FF69B4")
        self.place_at_grid(plate, 'B4', scale_factor=0.6)
        self.play(FadeIn(plate))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#7FFF00")
        self.place_at_grid(interference_label, 'C4', scale_factor=0.5)
        fringe = VGroup(*[Line(plate.get_left(), plate.get_right(), color="#FFFFFF", stroke_width=1) for _ in range(5)])
        fringe.arrange(DOWN, buff=0.1).move_to(plate)
        self.play(Create(fringe), Write(interference_label))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#FFA500")
        hologram_icon.set_color("#FFFFFF")
        self.place_at_grid(hologram_icon, 'C4', scale_factor=0.8)
        self.play(
            FadeOut(light_beams), FadeOut(fringe), FadeOut(plate), FadeOut(interference_label),
            FadeIn(hologram_icon)
        )
        self.wait(2)
