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
        self.setup_layout("Holographic Reconstruction", [
            "The hologram acts as a grating.",
            "Diffracted light recreates original wavefronts.",
            "Depth emerges from phase reconstruction.",
            "Illumination reveals the 3D illusion.",
            "Virtual objects appear in space."
        ])
        
        # Create elements with assets
        # Note: Fixed the file path for rabbit
        film = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/film.svg", color=WHITE)
        laser = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/laser.svg", color="#FF0000")
        camera = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/camera.svg", color=WHITE)
        rabbit = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/rabbit.svg", color="#00FFFF")
        
        rabbit_label = Text("Rabbit", font_size=24, color="#00FFFF").scale(0.7)

        # Place elements as instructed
        self.place_at_grid(film, 'D2', scale_factor=0.9)
        self.place_at_grid(rabbit, 'D5', scale_factor=0.75)
        self.place_in_area(rabbit_label, 'D5', 'D6', scale_factor=0.6)
        
        # Place laser and camera off-grid/initial positions
        laser.move_to(self.grid['D1']).scale(0.5)
        camera.move_to(self.grid['A6']).scale(0.8)
        
        beam = Line(start=laser.get_right(), end=film.get_left(), color="#FF0000")
        
        rabbit.set_opacity(0)
        rabbit_label.set_opacity(0)
        
        # --- Animations ---
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(self.lecture[0]), FadeIn(film))
        
        # === Animation for Lecture Line 2 ===
        self.play(FadeIn(self.lecture[1]), Create(beam), FadeIn(laser))
        self.lecture[0].set_color("#808080")
        
        # === Animation for Lecture Line 3 ===
        self.play(FadeIn(self.lecture[2]))
        self.lecture[1].set_color("#FF4500")
        
        # === Animation for Lecture Line 4 ===
        self.play(FadeIn(self.lecture[3]), FadeIn(rabbit), FadeIn(rabbit_label))
        self.lecture[2].set_color("#00CED1")
        
        # === Animation for Lecture Line 5 ===
        self.play(FadeIn(self.lecture[4]), FadeIn(camera))
        self.lecture[3].set_color("#FFD700")
        self.lecture[4].set_color("#FFD700")
