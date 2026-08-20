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
        self.setup_layout("The Holographic Reconstruction", [
            "Reference light hits the recorded pattern.",
            "The film acts like a diffraction grating.",
            "Light bends to recreate original wavefronts.",
            "The object appears to float in space.",
            "We have successfully reconstructed the 3D scene."
        ])
        
        # Load Assets
        # Note: SVGMobject for standard assets
        laser = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/laser.svg")
        hologram = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/hologram.svg")
        
        # Setup visual elements
        plate = Rectangle(width=0.5, height=3, color=WHITE, fill_opacity=0.3)
        self.place_at_grid(plate, "C5", scale_factor=0.7)
        
        # Labeling (B011, B020)
        plate_label = Text("Holographic Plate", color=WHITE).scale(0.7 * 0.75)
        plate_label.next_to(plate, UP, buff=0.2)
        
        # Grouping (B019, B037, B030)
        holographic_group = VGroup(plate, plate_label)
        self.place_in_area(holographic_group, 'B4', 'E6', scale_factor=0.5)
        
        beam = Line(start=LEFT*2, end=RIGHT*0, color="#00BFFF", stroke_width=4)
        
        # Sequence
        self.play(FadeIn(self.lecture))
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#00BFFF"), FadeIn(beam))
        self.place_at_grid(laser, "B4", scale_factor=0.7)
        self.play(FadeIn(laser), beam.animate.shift(RIGHT*2))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFD700"))
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFD700"))
        diffracted_rays = VGroup(*[Line(start=plate.get_right(), end=plate.get_right()+RIGHT*2+UP*(i-1), color="#FFD700") for i in range(3)])
        self.play(Create(diffracted_rays))
        
        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#FF4500"))
        # Using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/hologram.svg]
        self.place_at_grid(hologram, "E5", scale_factor=0.7)
        self.play(FadeIn(hologram))
        
        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#FF4500"))
        self.wait(2)
