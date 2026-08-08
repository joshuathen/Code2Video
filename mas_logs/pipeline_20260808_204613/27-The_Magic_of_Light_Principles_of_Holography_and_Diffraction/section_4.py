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
        self.setup_layout("The Holography Mechanism: Reconstruction", [
            "Reconstruction uses the reference beam.",
            "The hologram acts like a key.",
            "Original wavefronts emerge from diffraction.",
            "This reconstructs the 3D volume.",
            "Viewing angles shift perspective accurately."
        ])

        # Assets paths
        laser_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/laser.svg"
        hologram_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/hologram.svg"
        lens_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/lens.svg"

        # Load assets
        ref_beam = SVGMobject(laser_path)
        hologram = SVGMobject(hologram_path)
        lens = SVGMobject(lens_path)
        reconstructed_obj = Sphere(radius=0.5, color=RED).set_fill(RED, opacity=0.6)

        # Position assets using fixed grid logic per issues
        self.place_at_grid(ref_beam, 'B3', scale_factor=0.7)
        self.place_at_grid(hologram, 'D3', scale_factor=0.7)
        self.place_in_area(reconstructed_obj, 'B4', 'E5', scale_factor=0.6)
        self.place_at_grid(lens, 'D5', scale_factor=0.7)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.play(FadeIn(ref_beam))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(BLUE)
        self.play(FadeIn(hologram))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(GREEN)
        self.play(FadeIn(lens))

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(RED)
        self.play(FadeIn(reconstructed_obj))

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(ORANGE)
        self.play(reconstructed_obj.animate.shift(LEFT*0.5 + UP*0.2), run_time=2)
