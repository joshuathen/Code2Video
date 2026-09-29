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
            "Bats use chirps for echolocation.",
            "Short duration provides distance resolution.",
            "Frequency sweep captures target signature."
        ]
        self.setup_layout("Application: The Bat's Echolocation", lecture_lines)
        
        # Assets
        bat = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bat.svg")
        insect = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/insect.svg")
        
        self.place_at_grid(bat, 'B2', scale_factor=0.3)
        self.place_at_grid(insect, 'D5', scale_factor=0.3)
        self.add(bat, insect)
        
        # === Animation for Lecture Line 1 ===
        wave = ParametricFunction(
            lambda t: np.array([t, 0.5 * np.sin(10 * t) * np.exp(-2 * (t - 0.5)**2), 0]),
            t_range=[0, 1],
            color="#FF5733"
        )
        self.place_at_grid(wave, 'C2', scale_factor=0.8)
        self.play(Create(wave))
        self.lecture[0].set_color("#FF5733")

        # === Animation for Lecture Line 2 ===
        reflection = wave.copy().set_color(YELLOW)
        self.play(wave.animate.shift(RIGHT * 2.5), run_time=1.5)
        self.play(FadeIn(reflection))
        self.play(reflection.animate.shift(LEFT * 2.5), run_time=1.5)
        self.lecture[1].set_color("#FFD700")

        # === Animation for Lecture Line 3 ===
        line = Line(start=bat.get_center(), end=insect.get_center(), color=WHITE)
        delta_t = Text("Δt", font_size=20, color=WHITE)
        self.place_in_area(delta_t, 'D4', 'D5', scale_factor=0.7)
        
        self.play(Create(line), Write(delta_t))
        self.lecture[2].set_color("#00FFFF")
        self.wait(2)
