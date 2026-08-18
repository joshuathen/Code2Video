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
            "Fourier series maps signals to frequency coefficients.",
            "This is vital for MP3s and images.",
            "Our ears use this to interpret sound."
        ]
        self.setup_layout("Summary & Wrap-up", lecture_lines)
        
        # Elements
        waves = VGroup(
            *[FunctionGraph(lambda x: np.sin(k * x), x_range=[-2, 2], color=BLUE_D).shift(UP * 0.2 * k) for k in range(1, 4)]
        )
        # Using SVGMobject for the assets
        mp3_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/mp3.svg")
        ear_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ear.svg")
        
        # Positioning via grid
        self.place_at_grid(waves, 'B3', scale_factor=0.6)
        self.place_at_grid(mp3_icon, 'D4', scale_factor=0.6)
        self.place_at_grid(ear_icon, 'F4', scale_factor=0.6)
        
        mp3_label = Text("MP3/JPEG", font_size=20, color=YELLOW).next_to(mp3_icon, RIGHT)
        ear_label = Text("Ear/Cochlea", font_size=20, color=GREEN).next_to(ear_icon, RIGHT)

        # Animations
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFFFF"), run_time=1)
        self.play(FadeIn(waves))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFFF00"), run_time=1)
        self.play(FadeIn(mp3_icon), Write(mp3_label))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00FF00"), run_time=1)
        self.play(FadeIn(ear_icon), Write(ear_label))
        
        self.wait(2)
