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
        lecture_lines = [
            "Observe the squirrel gathering nuts over time.",
            "Derivative shows instantaneous nut-gathering speed.",
            "Integral calculates total nuts collected."
        ]
        self.setup_layout("Intuitive Hook: The Motion Analogy", lecture_lines)
        
        # Assets
        squirrel = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/squirrel.svg")
        nuts = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/nuts.svg")
        
        # Elements
        curve = FunctionGraph(lambda t: 0.3 * np.sin(t * 3) + 0.5, x_range=[0, 3], color=WHITE)
        label = Text("Position at time t", font_size=18, color=WHITE)
        tangent = Line(start=LEFT*0.5, end=RIGHT*0.5, color="#00FFFF", stroke_width=4)
        
        # Fix: Line 60: self.place_in_area(curve, 'B2', 'D5', scale_factor=0.6);
        self.place_in_area(curve, "B2", "D5", scale_factor=0.6)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FF00FF"))
        self.play(Create(curve))
        
        # Fix: Line 65: self.place_at_grid(point, 'C3', scale_factor=1.0) -> squirrel
        self.place_at_grid(squirrel, "C3", scale_factor=0.5)
        self.play(FadeIn(squirrel))
        
        # Fix: Line 67: self.place_at_grid(label, 'B4', scale_factor=0.7)
        self.place_at_grid(label, "B4", scale_factor=0.7)
        self.play(Write(label))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00FFFF"))
        # Fix: Line 72: self.place_at_grid(tangent, 'C3', scale_factor=0.7)
        self.place_at_grid(tangent, "C3", scale_factor=0.7)
        self.play(Create(tangent))
        
        # Place nuts
        self.place_at_grid(nuts, "E5", scale_factor=0.5)
        self.play(FadeIn(nuts))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFFF00"))
        area =  self.place_in_area(VMobject(), "C1", "E6")
        area.set_fill("#FFFF00", opacity=0.3)
        area.set_stroke(opacity=0)
        self.play(FadeIn(area))
        
        self.wait(2)
