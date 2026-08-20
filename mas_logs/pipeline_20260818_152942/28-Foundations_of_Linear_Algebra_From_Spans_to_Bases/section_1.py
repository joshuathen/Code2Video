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

class Section1Scene(TeachingScene):
    def construct(self):
        lecture_lines = ["Vectors act as instructions in space.", "We can scale these vectors.", "We can add vectors together."]
        self.setup_layout("Prerequisite: The Vector as an Instruction", lecture_lines)
        
        # Setup Axes
        axes = Axes(x_range=[-1, 5, 1], y_range=[-1, 5, 1], axis_config={"include_tip": True})
        self.place_in_area(axes, 'C2', 'D5', scale_factor=0.5)
        self.add(axes)

        # Assets
        pencil = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/pencil.svg")
        ruler = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ruler.svg")
        protractor = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/protractor.svg")

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFD700")
        vec = Vector([2, 3], color="#FFD700")
        vec.shift(axes.c2p(0, 0))
        pencil_icon = self.place_at_grid(pencil, 'A3', scale_factor=0.3)
        self.play(FadeIn(pencil_icon), Create(vec), run_time=1.5)
        
        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#32CD32")
        ruler_icon = self.place_at_grid(ruler, 'A4', scale_factor=0.3)
        scaled_vec = Vector([4, 6], color="#32CD32")
        scaled_vec.shift(axes.c2p(0, 0))
        self.play(FadeIn(ruler_icon), FadeIn(scaled_vec), run_time=1.5)
        
        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FF1493")
        protractor_icon = self.place_at_grid(protractor, 'A5', scale_factor=0.3)
        vec_add = Vector([2, 3], color="#FF1493")
        vec_add.shift(axes.c2p(2, 3))
        self.place_at_grid(vec_add, 'D4', scale_factor=0.7)
        self.play(FadeIn(protractor_icon), Create(vec_add), run_time=1.5)
        self.wait(2)
