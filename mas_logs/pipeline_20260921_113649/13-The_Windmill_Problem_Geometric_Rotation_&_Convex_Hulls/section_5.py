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
        lecture_lines = ["This concept aids convex hull algorithms.", "Geometry patterns reveal hidden structures.", "The windmill maps the entire plane."]
        self.setup_layout("Application and Conclusion", lecture_lines)
        
        # Asset Loading: Windmill SVG
        windmill_svg = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/windmill.svg"
        windmill = SVGMobject(windmill_svg)
        
        # Elements for animation
        points = VGroup(*[Dot(radius=0.05).move_to(self.grid['B4'] + RIGHT*0.5*np.cos(i) + UP*0.5*np.sin(i)) for i in range(8)])
        
        # Animation sequence
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFF00"))
        self.play(FadeIn(points))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFFFFF"))
        self.place_at_grid(windmill, 'B5', scale_factor=0.6)
        self.play(FadeIn(windmill))
        
        structure_label = Text("Structure: predictable patterns", font_size=20, color=WHITE)
        self.place_at_grid(structure_label, 'D5', scale_factor=0.8)
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FF00FF"))
        text_summary = Text("Windmill mapping", font_size=20, color=WHITE)
        self.place_in_area(text_summary, 'D4', 'F6', scale_factor=0.7)
        self.play(FadeIn(text_summary), FadeIn(structure_label))
        
        self.wait(1)
        self.play(FadeOut(Group(windmill, points, text_summary, structure_label, self.lecture, self.title)))
