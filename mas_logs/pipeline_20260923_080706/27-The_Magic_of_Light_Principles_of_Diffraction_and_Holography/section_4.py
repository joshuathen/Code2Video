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
        self.setup_layout("Reconstruction: Decoding the 3D Image", [
            "Holograms act like diffraction gratings.",
            "Coherent light reconstructs original fronts.",
            "This projects images into space.",
            "The 3D scene appears reconstructed.",
            "Complex data becomes visible light."
        ])
        
        laser = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/laser.svg")
        hologram = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/hologram.svg")
        reconstructed_wave = Circle(radius=0.5, color=BLUE)
        obj_3d = Cube(side_length=0.5, fill_opacity=0.5, color=YELLOW)
        label_3d = Text("3D_Reconstruction", font_size=20, color=GREEN)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFFFF")
        self.place_at_grid(laser, 'B1', scale_factor=0.5)
        self.place_at_grid(hologram, 'B3', scale_factor=0.8)
        self.play(FadeIn(laser), FadeIn(hologram))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#00FFFF")
        self.place_at_grid(reconstructed_wave, 'B5', scale_factor=1.0)
        self.play(Create(reconstructed_wave))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FF00FF")
        self.play(FadeIn(obj_3d))

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#FFFF00")
        self.place_at_grid(obj_3d, 'E5', scale_factor=0.7)
        self.play(obj_3d.animate.set_color("#FFFF00"))

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#00FF00")
        self.place_at_grid(label_3d, 'F5', scale_factor=0.8)
        self.play(FadeIn(label_3d))
        self.wait(2)
