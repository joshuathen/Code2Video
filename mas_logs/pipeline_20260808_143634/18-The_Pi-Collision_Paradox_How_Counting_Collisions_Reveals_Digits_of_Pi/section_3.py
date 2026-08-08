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
        lecture_lines = [
            "Mass ratios reveal digits of Pi.",
            "Ratio 100 yields 31 collisions.",
            "Ratio 10,000 yields 314 collisions.",
            "Ratio 1,000,000 yields 3,141 collisions.",
            "Collisions count digits of Pi."
        ]
        self.setup_layout("The Scaling Connection: Linking Physics to Geometry", lecture_lines)
        
        # Persistent mobjects
        counter = Text("0", font_size=48, color=YELLOW)
        self.place_at_grid(counter, 'B5', scale_factor=0.8)
        
        circle = Circle(radius=1.5, color=WHITE)
        self.place_at_grid(circle, 'E4', scale_factor=0.9)
        
        # Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/blocks.svg
        try:
            blocks_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/blocks.svg")
        except:
            blocks_icon = Square(color=RED)
            
        group_combined = VGroup(counter, blocks_icon)
        self.place_in_area(group_combined, 'A2', 'F2', scale_factor=0.8)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(YELLOW))
        counter.become(Text("31", font_size=48, color=YELLOW).move_to(counter.get_center()))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(YELLOW))
        counter.become(Text("314", font_size=48, color=YELLOW).move_to(counter.get_center()))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(YELLOW))
        counter.become(Text("3141", font_size=48, color=YELLOW).move_to(counter.get_center()))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(YELLOW))
        self.play(Create(circle))
        self.wait(2)
