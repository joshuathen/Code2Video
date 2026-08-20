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
        lecture_lines = ["We measure distance by simple magnitude.", "Standard convergence means points get closer.", "A hiker approaches a signpost steadily."]
        self.setup_layout("Prerequisite: The Usual Notion of Convergence", lecture_lines)
        
        # Elements
        line = NumberLine(x_range=[0, 2, 0.5], length=5)
        hiker = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/hiker.svg")
        signpost = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/signpost.svg")
        
        # Positioning based on VideoCritic feedback
        self.place_in_area(line, 'D2', 'D5', scale_factor=0.8)
        self.place_at_grid(hiker, 'B3', scale_factor=0.7)
        self.place_at_grid(signpost, 'B5', scale_factor=0.7)
        
        # Labels
        label_h = Text("Hiker", font_size=16)
        label_s = Text("Limit", font_size=16)
        self.place_at_grid(label_h, 'A3', scale_factor=0.6)
        self.place_at_grid(label_s, 'A5', scale_factor=0.6)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.play(Create(line), FadeIn(hiker), FadeIn(signpost), FadeIn(label_h), FadeIn(label_s))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FF00FF"))
        # Visualize convergence
        target = self.grid["B5"]
        self.play(hiker.animate.move_to(target), run_time=2)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00FFFF"))
        self.play(Indicate(signpost))
        self.wait(1)
