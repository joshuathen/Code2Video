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
            "Energy enters large and dissipates at η.",
            "Constants enable modeling without tracking atoms.",
            "Turbulence models help design efficient systems."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Atmosphere and pipe flow representation
        atm = Rectangle(width=1.5, height=1.5, color=BLUE).add(Text("Atmosphere", font_size=12))
        pipe = Rectangle(width=1.5, height=1.5, color=YELLOW).add(Text("Pipe Flow", font_size=12))
        
        side_by_side = VGroup(atm, pipe).arrange(RIGHT, buff=0.3)
        # Fix VideoCritic 38 & 40: Repositioning group to align better and reduce footprint
        self.place_in_area(side_by_side, 'B3', 'C6', scale_factor=0.6)
        
        self.play(Create(side_by_side))
        self.lecture[0].set_color(BLUE)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Summarize K41 utility
        # Fix VideoCritic 39: Repositioning checkmark higher
        checkmark = Tex(r"$\checkmark$", color="#2ECC71").scale(1.5)
        self.place_at_grid(checkmark, 'D4', scale_factor=0.4)
        
        self.play(FadeIn(checkmark))
        self.lecture[1].set_color("#2ECC71")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/falcon.svg
        falcon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/falcon.svg", color="#2ECC71")
        self.place_at_grid(falcon, 'E4', scale_factor=0.5)
        
        self.play(FadeIn(falcon))
        self.lecture[2].set_color("#2ECC71")
        self.wait(2)
        
        self.play(FadeOut(side_by_side), FadeOut(checkmark), FadeOut(falcon))
        self.wait(1)
