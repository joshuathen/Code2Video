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
        self.setup_layout("Summary and Real-world Application", [
            "Diffraction and interference enable 3D imaging.",
            "Lasers provide the necessary coherent light.",
            "Holography secures data and visual information."
        ])
        
        # Elements
        img_3d = Circle(radius=0.5, color=BLUE).add(Text("3D", font_size=12))
        laser = Line(start=LEFT, end=RIGHT, color=RED).add(Text("Laser", font_size=12, color=RED).next_to(ORIGIN, UP))
        banknote = Rectangle(width=1.5, height=0.8, color=GREEN)
        banknote_text = Text("Holo-Security", font_size=12, color=GREEN).move_to(banknote.get_center())
        banknote_group = VGroup(banknote, banknote_text)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(BLUE)
        self.place_at_grid(img_3d, 'B1', scale_factor=0.6)
        self.play(Create(img_3d))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(RED)
        self.place_in_area(laser, 'C1', 'C3', scale_factor=0.7)
        self.play(Create(laser))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(GREEN)
        self.place_in_area(banknote_group, 'D1', 'E3', scale_factor=0.7)
        self.play(FadeIn(banknote_group))
        self.wait(2)
