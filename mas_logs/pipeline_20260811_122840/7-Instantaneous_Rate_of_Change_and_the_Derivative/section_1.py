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
        self.setup_layout("The Hook: From Average to Instantaneous", [
            "Cheetah runs 100m in 5s, average speed 20m/s.",
            "How fast is the cheetah at exactly 2s?",
            "Shrink the time interval toward zero to see."
        ])
        
        self.lecture.set_opacity(0)

        # Assets
        cheetah = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cheetah.svg", color="#FF4500")
        clock = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/clock.svg", color="#00BFFF")
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_opacity(1)
        path = Line(start=self.grid['C4'], end=self.grid['C6'], color=GREY)
        self.add(path)
        self.place_in_area(cheetah, 'B2', 'B3', scale_factor=0.5)
        self.play(FadeIn(cheetah), run_time=0.5)
        self.play(cheetah.animate.move_to(self.grid['C6']), run_time=2, rate_func=linear)
        self.lecture[0].set_color("#FF4500")

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_opacity(1)
        point_at_2s = Dot(color="#00BFFF")
        self.place_at_grid(point_at_2s, 'B5', scale_factor=1.0)
        self.place_at_grid(clock, 'E5', scale_factor=0.5)
        self.play(FadeIn(point_at_2s), FadeIn(clock), run_time=0.5)
        self.lecture[1].set_color("#00BFFF")

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_opacity(1)
        interval = Line(start=self.grid['B4'], end=self.grid['B6'], color="#32CD32")
        self.play(Create(interval), run_time=1)
        self.play(interval.animate.scale(0.1).move_to(self.grid['B5']), run_time=2)
        self.lecture[2].set_color("#32CD32")
        self.wait(1)
