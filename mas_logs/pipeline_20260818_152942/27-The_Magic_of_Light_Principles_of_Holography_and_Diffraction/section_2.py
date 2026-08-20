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

class Section2Scene(TeachingScene):
    def construct(self):
        self.setup_layout("The Mechanism of Holography: Recording Phase", [
            "Photography captures only light intensity.",
            "Holography records amplitude and phase.",
            "A reference beam creates interference."
        ])
        
        # Objects using Assets
        laser = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/laser.svg", color=RED)
        beam_splitter = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/mirror.svg", color=BLUE)
        mirror = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/mirror.svg", color=WHITE)
        object_dot = Dot(color=YELLOW)
        plate = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/plate.svg", color=GREY)
        
        # Assemble apparatus group for area placement
        apparatus = VGroup(laser, beam_splitter, mirror, object_dot, plate)
        
        # Positioning (applying requested adjustments)
        # Using self.place_in_area for the group
        self.place_in_area(apparatus, 'C4', 'F6', scale_factor=0.8)
        # Refine specific positions
        self.place_at_grid(beam_splitter, 'D3', scale_factor=0.9)
        self.place_at_grid(plate, 'D6', scale_factor=1.0)
        
        # Add to scene
        self.add(apparatus)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        path1 = Line(laser.get_center(), beam_splitter.get_center(), color=RED)
        self.play(Create(path1))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(BLUE))
        path2_obj = Line(beam_splitter.get_center(), object_dot.get_center(), color=BLUE)
        path2_plate = Line(object_dot.get_center(), plate.get_center(), color=BLUE)
        self.play(Create(path2_obj), Create(path2_plate))
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(GREEN))
        path_ref = Line(beam_splitter.get_center(), mirror.get_center(), color=GREEN)
        path_ref_to_plate = Line(mirror.get_center(), plate.get_center(), color=GREEN)
        self.play(Create(path_ref), Create(path_ref_to_plate))
        
        self.wait(2)
