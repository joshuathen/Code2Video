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

class Section3Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Decoding the Time Signature", [
            "Time signatures act as maps.", 
            "The top number counts beats.", 
            "The bottom shows note values.", 
            "Whole notes split into parts.", 
            "Four beats fill one measure."
        ])
        
        # Elements
        fraction = MathTex(r"{4 \over 4}", font_size=96)
        top_num = fraction[0][0]
        bottom_num = fraction[0][2]
        
        map_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/map.svg")
        self.place_at_grid(map_icon, 'B3', scale_factor=0.5)
        
        self.place_at_grid(fraction, 'C2', scale_factor=1.2)

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(fraction), FadeIn(map_icon))
        self.play(self.lecture[0].animate.set_color(YELLOW))

        # === Animation for Lecture Line 2 ===
        self.play(top_num.animate.set_color("#FF0000"))
        self.play(self.lecture[1].animate.set_color("#FF0000"))

        # === Animation for Lecture Line 3 ===
        self.play(bottom_num.animate.set_color("#0000FF"))
        self.play(self.lecture[2].animate.set_color("#0000FF"))

        # === Animation for Lecture Line 4 ===
        pie = VGroup(
            Sector(radius=1.0, angle=PI/2, start_angle=0, color="#FFCC00"),
            Sector(radius=1.0, angle=PI/2, start_angle=PI/2, color="#FFCC00"),
            Sector(radius=1.0, angle=PI/2, start_angle=PI, color="#FFCC00"),
            Sector(radius=1.0, angle=PI/2, start_angle=3*PI/2, color="#FFCC00")
        )
        self.place_at_grid(pie, 'D3', scale_factor=0.6)
        self.play(Create(pie))
        self.play(self.lecture[3].animate.set_color(WHITE))

        # === Animation for Lecture Line 5 ===
        self.play(fraction.animate.set_color(WHITE))
        self.play(self.lecture[4].animate.set_color(GREEN))
        self.wait(2)
