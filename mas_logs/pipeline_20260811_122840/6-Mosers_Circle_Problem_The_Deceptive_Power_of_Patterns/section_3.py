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
        self.setup_layout("The Counter-Intuitive Reality", ["Now check 6 points carefully.", "The result is 31, not 32.", "Patterns can be deceiving traps."])
        
        # === Animation for Lecture Line 1 ===
        circle = Circle(radius=1.5, color=WHITE)
        points = [Dot(circle.point_from_proportion(i/6)) for i in range(6)]
        points_group = VGroup(*points)
        diagram_group = VGroup(circle, points_group)
        
        # Use Asset
        knife = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/knife.svg", color=WHITE)
        
        self.place_in_area(diagram_group, 'A3', 'C6', scale_factor=0.8)
        self.play(Create(circle), Write(points_group))
        
        # Adding the asset
        self.place_at_grid(knife, 'A4', scale_factor=0.5)
        self.play(FadeIn(knife))
        
        self.lecture[0].set_color("#FFFFFF")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        lines = VGroup()
        for i in range(6):
            for j in range(i + 1, 6):
                lines.add(Line(points[i].get_center(), points[j].get_center(), color=BLUE, stroke_width=2))
        
        self.play(Create(lines), run_time=2)
        
        result_text = Text("31 Regions", font_size=32, color=RED)
        self.place_at_grid(result_text, 'E4', scale_factor=0.9)
        self.play(Write(result_text))
        self.lecture[1].set_color("#FF4500")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        arrow_icon = Arrow(start=UP, end=DOWN, color="#00CED1")
        self.place_at_grid(arrow_icon, 'F4', scale_factor=0.7)
        self.add(arrow_icon)
        
        self.play(Indicate(result_text), Indicate(arrow_icon))
        self.lecture[2].set_color("#00CED1")
        self.wait(2)
