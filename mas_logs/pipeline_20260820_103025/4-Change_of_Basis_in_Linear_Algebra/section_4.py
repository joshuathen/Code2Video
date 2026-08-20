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
        self.setup_layout("Practical Application & Summary", [
            "Choosing a smart basis simplifies complex motion.",
            "Rotation is just scaling in the right basis.",
            "Change of basis reveals the simplest perspective."
        ])
        
        # Elements
        gear = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/gear.svg")
        math_box = MathTex(r"T = \begin{bmatrix} \lambda_1 & 0 \\ 0 & \lambda_2 \end{bmatrix}", font_size=36)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#F1C40F")
        # Ensure column buffer as requested
        self.place_in_area(gear, 'B4', 'D6', scale_factor=0.7)
        self.play(FadeIn(gear))

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#3498DB")
        # Gear rotation
        gear.add_updater(lambda m, dt: m.rotate(dt * PI / 4))
        self.wait(2)
        gear.remove_updater(None)
        
        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#2ECC71")
        
        self.place_in_area(math_box, 'C4', 'E6', scale_factor=0.9)
        math_box.set_color("#2ECC71")
        
        self.play(FadeIn(math_box))
        self.play(Flash(math_box, color="#2ECC71"))
        self.wait(2)
