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
        self.setup_layout("Real-World Application: Torque", [
            "Torque is the cross product.",
            "Force applied to lever arm vectors.",
            "Determines the direction of rotation.",
            "Calculated using the cross product.",
            "Applied in real-world mechanical systems."
        ])
        
        # Define objects
        lever = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/lever.svg")
        gear = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/gear.svg")
        force = Arrow(ORIGIN, 0.7*UP, color="#FF0000", buff=0)
        torque = Arrow(ORIGIN, 1*OUT, color="#00FF00", buff=0)
        
        vector_animation = VGroup(lever, force, torque)
        force_label = Text("F", color=YELLOW)
        calculation_highlight = Circle(radius=0.3, color=ORANGE)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.place_at_grid(lever, "B2", scale_factor=0.5)
        self.play(FadeIn(lever))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FF0000"))
        self.place_in_area(vector_animation, "D1", "F6", scale_factor=0.9)
        self.place_at_grid(force_label, "D2", scale_factor=0.7)
        self.play(GrowArrow(force), FadeIn(force_label))
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFFFFF"))
        # Rotation indication
        self.play(Rotate(lever, angle=PI/4, about_point=lever.get_center()))
        
        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#00FF00"))
        self.place_at_grid(calculation_highlight, "E4", scale_factor=0.8)
        self.play(GrowArrow(torque), Create(calculation_highlight))
        
        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#FFA500"))
        self.place_at_grid(gear, "C5", scale_factor=0.5)
        self.play(FadeIn(gear), Rotate(gear, angle=2*PI))
        self.wait(1)
