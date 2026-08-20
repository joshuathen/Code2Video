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

class Section2Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Prerequisite Intuition: Kinetic vs. Potential Energy", 
                          ["Gravity converts potential energy into kinetic energy.", 
                           "Steeper descents yield higher speeds earlier.", 
                           "This speed boost helps traverse the later path."])
        
        # Define elements
        pot_label = Text("Potential", color="#00FF00", font_size=24)
        kin_label = Text("Kinetic", color="#FF0000", font_size=24)
        
        pot_bar = Rectangle(width=0.5, height=3, fill_opacity=1, color="#00FF00")
        kin_bar = Rectangle(width=0.5, height=0.1, fill_opacity=1, color="#FF0000")
        
        # Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/ball.svg
        ball = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ball.svg")
        
        # Applying requested fixes
        self.place_at_grid(pot_label, "B2", scale_factor=0.6)
        self.place_at_grid(kin_label, "B5", scale_factor=0.6)
        self.place_in_area(pot_bar, "C2", "D2", scale_factor=0.4)
        self.place_in_area(kin_bar, "C5", "D5", scale_factor=0.4)
        self.place_at_grid(ball, "A3", scale_factor=0.5)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFF00"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFFF00"))
        
        # Animate ball descent and bars
        self.play(
            ball.animate.move_to(self.grid["F3"]),
            pot_bar.animate.set_height(0.1),
            kin_bar.animate.set_height(3),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFFF00"))
        self.wait(2)
