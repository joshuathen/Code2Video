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
            "Place n points on a circle's edge.",
            "Connect every pair with straight chords.",
            "Count all resulting inner regions.",
            "We create slices like a pizza.",
            "How many regions for n points?"
        ]
        self.setup_layout("Moser's Circle Problem", lecture_lines)
        
        # Geometry container
        geometry_container = VGroup()
        
        # Pizza icon
        pizza = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/pizza.svg", color=WHITE)
        self.place_at_grid(pizza, 'A6', scale_factor=0.3)
        self.add(pizza)
        
        circle = Circle(radius=1.2, color=WHITE)
        point_group = VGroup()
        
        # Fixes from issues
        self.place_at_grid(circle, 'C4', scale_factor=0.9)
        self.place_at_grid(point_group, 'C4', scale_factor=0.9)
        geometry_container.add(circle, point_group)
        self.place_in_area(geometry_container, 'B3', 'E5', scale_factor=0.75)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(circle), self.lecture[0].animate.set_color("#FFFFFF"))
        point1 = Dot(circle.point_at_angle(PI/2), color=YELLOW)
        p1_label = Text("P1", font_size=16).next_to(point1, UP, buff=0.1)
        r1_label = Text("R1", font_size=16).move_to(circle.get_center())
        point_group.add(point1, p1_label, r1_label)
        self.play(FadeIn(point_group))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFFF00"))
        point2 = Dot(circle.point_at_angle(PI/2 + 2*PI/3), color=YELLOW)
        chord1 = Line(point1.get_center(), point2.get_center(), color=YELLOW)
        point_group.add(point2, chord1)
        self.play(FadeIn(point2), Create(chord1))
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00FFFF"))
        point3 = Dot(circle.point_at_angle(PI/2 - 2*PI/3), color=YELLOW)
        chord2 = Line(point1.get_center(), point3.get_center(), color="#00FFFF")
        chord3 = Line(point2.get_center(), point3.get_center(), color="#00FFFF")
        point_group.add(point3, chord2, chord3)
        self.play(FadeIn(point3), Create(chord2), Create(chord3))
        
        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#FF00FF"))
        point4 = Dot(circle.point_at_angle(0), color=YELLOW)
        chords4 = VGroup(
            Line(point4.get_center(), point1.get_center(), color="#FF00FF"),
            Line(point4.get_center(), point2.get_center(), color="#FF00FF"),
            Line(point4.get_center(), point3.get_center(), color="#FF00FF")
        )
        point_group.add(point4, chords4)
        self.play(FadeIn(point4), Create(chords4))

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#FFFFFF"))
        self.wait(2)
