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
        lecture_lines = [
            "Gradient descent is our hiker's path.",
            "We measure slope beneath our feet.",
            "We take small steps downhill daily.",
            "Repeat this until the valley floor.",
            "Weights settle into optimal configurations."
        ]
        self.setup_layout("Backpropagation: Stepping Down the Gradient", lecture_lines)
        
        # Create elements
        mountain = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/mountain.svg", color=BLUE)
        hiker = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/hiker.svg")
        vector = Arrow(ORIGIN, RIGHT * 0.5 + DOWN * 0.5, color=YELLOW)
        path = VGroup()
        
        # Place elements
        self.place_in_area(mountain, 'A2', 'E4', scale_factor=0.7)
        self.place_at_grid(hiker, 'C3', scale_factor=0.6)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFFFF")
        self.play(FadeIn(mountain), FadeIn(hiker))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFFF00")
        self.place_at_grid(vector, 'D4', scale_factor=0.7)
        self.play(GrowArrow(vector))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FF0000")
        target_pos = self.grid['D5']
        hiker_path = Line(hiker.get_center(), target_pos, color=RED)
        self.play(
            hiker.animate.move_to(target_pos),
            Create(hiker_path),
            FadeOut(vector)
        )

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#FF0000")
        final_pos = self.grid['E6']
        final_path = Line(hiker.get_center(), final_pos, color=RED)
        self.play(
            hiker.animate.move_to(final_pos),
            Create(final_path)
        )

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#00FF00")
        self.wait(1)
