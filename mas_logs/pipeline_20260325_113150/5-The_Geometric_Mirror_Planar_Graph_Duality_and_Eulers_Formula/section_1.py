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
        # Initialize layout with the specified title and lecture lines
        title_text = "Prerequisite: Defining the Planar Canvas"
        lecture_lines = [
            "Planar graphs have vertices, edges, and distinct regions called faces.",
            "This square creates one inner face and one outer face.",
            "We add a curved edge to avoid crossing other edges."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Positioning variables based on the 6x6 grid
        # Square corners
        v1_pos = self.grid["B2"]
        v2_pos = self.grid["B5"]
        v3_pos = self.grid["E5"]
        v4_pos = self.grid["E2"]

        # === Animation for Lecture Line 1 ===
        # Planar graphs have vertices, edges, and distinct regions called faces.
        self.play(self.lecture[0].animate.set_color("#00FF00"))
        
        # Create vertices (Green)
        v1 = Dot(v1_pos, color="#00FF00")
        v2 = Dot(v2_pos, color="#00FF00")
        v3 = Dot(v3_pos, color="#00FF00")
        v4 = Dot(v4_pos, color="#00FF00")
        vertices = VGroup(v1, v2, v3, v4)
        
        # Create edges (White)
        e1 = Line(v1_pos, v2_pos, color="#FFFFFF")
        e2 = Line(v2_pos, v3_pos, color="#FFFFFF")
        e3 = Line(v3_pos, v4_pos, color="#FFFFFF")
        e4 = Line(v4_pos, v1_pos, color="#FFFFFF")
        edges = VGroup(e1, e2, e3, e4)
        
        self.play(
            Create(vertices),
            Create(edges),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # This square creates one inner face and one outer face.
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color("#FFFF00")
        )
        
        # Highlight and label Face 1 (#FFFF00) inside the square
        face1_highlight = Polygon(v1_pos, v2_pos, v3_pos, v4_pos, 
                                  fill_color="#FFFF00", fill_opacity=0.3, stroke_width=0)
        face1_label = Text("Face 1", font_size=24, color="#FFFF00")
        # Issue 22 Fix: Move label to bottom-left quadrant to avoid crossing edge
        self.place_in_area(face1_label, 'D2', 'E3', scale_factor=0.8)
        
        # Highlight and label Face 2 (#FFA500) in the infinite area outside
        face2_label = Text("Face 2 (Outer)", font_size=24, color="#FFA500")
        # Issue 21 Fix: Center in a wider area to avoid clipping
        self.place_in_area(face2_label, 'A4', 'A6', scale_factor=0.8)
        
        # A dashed boundary to visually represent the concept of the "outer" face
        face2_boundary = DashedVMobject(Rectangle(width=5.0, height=5.0, color="#FFA500"))
        # Issue 23 Fix: Scale down to avoid edge-of-screen cramping
        self.place_in_area(face2_boundary, 'A1', 'F6', scale_factor=0.9)

        self.play(
            FadeIn(face1_highlight),
            Write(face1_label),
            run_time=1
        )
        self.play(
            Create(face2_boundary),
            Write(face2_label),
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # We add a curved edge to avoid crossing other edges.
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color("#FFFFFF")
        )
        
        # Draw a 5th edge (#FFFFFF) curved outside the square to demonstrate planarity without crossings.
        # Connecting top-left (v1) to bottom-right (v3) with a curve that stays outside the square bounds.
        e5 = ArcBetweenPoints(v1_pos, v3_pos, angle=-TAU/2, color="#FFFFFF")
        
        self.play(Create(e5), run_time=1.5)
        self.wait(2)
