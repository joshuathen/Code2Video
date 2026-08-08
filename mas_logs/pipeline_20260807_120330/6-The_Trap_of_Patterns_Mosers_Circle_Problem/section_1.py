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
        # Title and Lecture Lines from storyboard
        title = "The Hook: A Game of Slicing"
        lines = [
            "Imagine placing points along a circle's edge.",
            "Connect every point to every other point with lines.",
            "How many regions can we divide this circle into?"
        ]
        
        self.setup_layout(title, lines)
        
        # Colors
        COLOR_CIRCLE = "#FFFFFF"
        COLOR_POINT = "#FFFF00"
        COLOR_CHORD = "#FFFF00"
        COLOR_GLOW = "#ADD8E6"
        COLOR_HIGHLIGHT = "#FFFF00"

        # === Animation for Lecture Line 1 ===
        # Imagine placing points along a circle's edge.
        self.lecture[0].set_color(COLOR_HIGHLIGHT)
        
        # Define circle - Applied Fix for Issue 22
        # self.place_in_area(circle, 'A2', 'F6', scale_factor=0.8)
        circle = Circle(radius=2.2, color=COLOR_CIRCLE)
        self.place_in_area(circle, "A2", "F6", scale_factor=0.8)
        
        # Non-uniform angles (B059) to avoid center and symmetry
        angles = [35 * DEGREES, 155 * DEGREES, 260 * DEGREES, 335 * DEGREES]
        points = [Dot(circle.point_at_angle(a), color=COLOR_POINT, radius=0.1) for a in angles]
        
        self.play(Create(circle))
        self.play(FadeIn(points[0]), FadeIn(points[1]))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Connect every point to every other point with lines.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(COLOR_HIGHLIGHT)
        
        # Chord for 2 points
        chord1 = Line(points[0].get_center(), points[1].get_center(), color=COLOR_CHORD)
        self.play(Create(chord1))
        
        # Glow regions for 2 points (approximate using semi-transparent fill)
        glow2 = circle.copy().set_fill(COLOR_GLOW, opacity=0.3).set_stroke(width=0)
        self.play(FadeIn(glow2))
        self.play(FadeOut(glow2))
        
        # Add 3rd point and its chords
        self.play(FadeIn(points[2]))
        chord2 = Line(points[0].get_center(), points[2].get_center(), color=COLOR_CHORD)
        chord3 = Line(points[1].get_center(), points[2].get_center(), color=COLOR_CHORD)
        self.play(Create(VGroup(chord2, chord3)))
        
        # Glow 4 regions (approximate)
        glow3 = circle.copy().set_fill(COLOR_GLOW, opacity=0.4).set_stroke(width=0)
        self.play(FadeIn(glow3))
        self.play(FadeOut(glow3))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # How many regions can we divide this circle into?
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(COLOR_HIGHLIGHT)
        
        # Add 4th point and its chords
        self.play(FadeIn(points[3]))
        chord4 = Line(points[0].get_center(), points[3].get_center(), color=COLOR_CHORD)
        chord5 = Line(points[1].get_center(), points[3].get_center(), color=COLOR_CHORD)
        chord6 = Line(points[2].get_center(), points[3].get_center(), color=COLOR_CHORD)
        self.play(Create(VGroup(chord4, chord5, chord6)))
        
        # Highlight 8 regions - Flash and Pulse
        glow4 = circle.copy().set_fill(COLOR_GLOW, opacity=0.5).set_stroke(width=0)
        self.play(FadeIn(glow4))
        self.play(Flash(circle, color=COLOR_GLOW, line_length=0.5, num_lines=12))
        self.play(FadeOut(glow4))
        
        # Final Rule Emphasis: "Connect every point to every other point."
        self.lecture[2].set_color(WHITE)
        self.lecture[1].set_color(COLOR_HIGHLIGHT)
        self.wait(2)
        
        # Clear highlights
        self.play(self.lecture[1].animate.set_color(WHITE))
        self.wait(1)
