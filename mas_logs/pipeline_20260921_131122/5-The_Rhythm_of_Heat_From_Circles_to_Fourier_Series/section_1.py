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
            "Circular motion creates simple harmonic waves.",
            "A point rotates at a constant speed.",
            "The shadow traces a sine wave.",
            "This projection links angles to waves.",
            "Frequency defines the rotation speed."
        ]
        self.setup_layout("Prerequisite: The Geometry of Oscillation", lecture_lines)
        
        # Mobjects
        circle = Circle(radius=1, color=WHITE)
        pendulum_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/pendulum.svg", color=WHITE).scale(0.5)
        unit_circle_group = VGroup(circle, pendulum_icon).arrange(DOWN)
        
        # Incorporating feedback: Positioning with place_in_area
        self.place_in_area(unit_circle_group, 'B3', 'D6', scale_factor=0.65)
        label_circle = Text("Unit Circle", font_size=20, color=WHITE)
        label_circle.next_to(unit_circle_group, UP)
        
        point_p = Dot(circle.point_at_angle(0), color="#FF5733")
        label_p = Text("P", font_size=18, color="#FF5733")
        label_p.next_to(point_p, UR, buff=0.1)
        
        # Trace path (arc)
        path = Arc(radius=1, start_angle=0, angle=0, color="#FF5733")
        
        # Wave plotting
        metronome_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/metronome.svg", color="#33FF57").scale(0.5)
        wave_line = Axes(x_range=[0, 4, 1], y_range=[-1.5, 1.5, 1], axis_config={"include_numbers": False}).scale(0.5)
        wave_group = VGroup(wave_line, metronome_icon).arrange(DOWN)
        self.place_at_grid(wave_group, 'E5', scale_factor=0.7)
        
        # === Animation for Lecture Line 1 ===
        self.play(Create(circle), Write(pendulum_icon), Write(label_circle))
        self.lecture[0].set_color("#FFFFFF")

        # === Animation for Lecture Line 2 ===
        self.play(Create(point_p), Write(label_p))
        self.lecture[1].set_color("#FF5733")

        # === Animation for Lecture Line 3 ===
        self.play(Create(path), Rotate(point_p, angle=2*PI, about_point=circle.get_center(), rate_func=linear))
        self.lecture[2].set_color("#FF5733")

        # === Animation for Lecture Line 4 ===
        # Propose linking the wave projection
        self.lecture[3].set_color("#33FF57")

        # === Animation for Lecture Line 5 ===
        self.play(Create(wave_group))
        self.lecture[4].set_color("#33FF57")
        
        self.wait(2)
