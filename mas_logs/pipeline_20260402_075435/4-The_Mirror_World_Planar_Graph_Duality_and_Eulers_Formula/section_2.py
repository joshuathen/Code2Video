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
        # Setup layout with specific title and lines
        title = "Euler’s Magic Rule: V - E + F = 2"
        lines = [
            "Euler’s formula connects vertices, edges, and faces.",
            "For connected planar graphs, V minus E plus F is two.",
            "A square has four vertices, four edges, two faces.",
            "Add a diagonal edge; one face splits into two.",
            "The formula stays: four minus five plus three is two."
        ]
        self.setup_layout(title, lines)

        # === Animation for Lecture Line 1 ===
        # Write V - E + F = 2 in bold white (#FFFFFF) text.
        self.lecture[0].set_color(YELLOW)
        formula_main = Text("V - E + F = 2", color="#FFFFFF", font_size=60)
        self.place_in_area(formula_main, 'A2', 'B5')
        
        self.play(Write(formula_main))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight logic for connected planar graphs
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        self.play(formula_main.animate.scale(1.1), run_time=0.5)
        self.play(formula_main.animate.scale(1/1.1), run_time=0.5)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Show a square graph with 4 green (#00FF00) vertices and 4 cyan (#00FFFF) edges.
        # Highlight the interior and exterior faces (F=2) in magenta (#FF00FF).
        # Display calculation 4 - 4 + 2 = 2 below the graph.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)

        # Vertices
        v1 = Dot(color="#00FF00")
        v2 = Dot(color="#00FF00")
        v3 = Dot(color="#00FF00")
        v4 = Dot(color="#00FF00")
        self.place_at_grid(v1, 'C2')
        self.place_at_grid(v2, 'C5')
        self.place_at_grid(v3, 'E5')
        self.place_at_grid(v4, 'E2')
        verts = VGroup(v1, v2, v3, v4)

        # Edges
        e1 = Line(v1.get_center(), v2.get_center(), color="#00FFFF")
        e2 = Line(v2.get_center(), v3.get_center(), color="#00FFFF")
        e3 = Line(v3.get_center(), v4.get_center(), color="#00FFFF")
        e4 = Line(v4.get_center(), v1.get_center(), color="#00FFFF")
        edges = VGroup(e1, e2, e3, e4)

        # Faces
        # Interior Face
        face_int = Polygon(v1.get_center(), v2.get_center(), v3.get_center(), v4.get_center(), 
                          fill_opacity=0.3, fill_color="#FF00FF", stroke_width=0)
        label_f1 = Text("F1", font_size=18, color="#FF00FF")
        self.place_in_area(label_f1, 'C3', 'D4')
        
        # Exterior Face (represented by a surrounding frame)
        face_ext_rect = Rectangle(width=4.5, height=3.5, color="#FF00FF", stroke_width=2, fill_opacity=0.1)
        self.place_in_area(face_ext_rect, 'C1', 'E6')
        label_f_ext = Text("F2 (Ext)", font_size=18, color="#FF00FF")
        # Fixed (Issue 34): Move to D6 to avoid corner/edge clipping
        self.place_at_grid(label_f_ext, 'D6')

        # Calculation
        calc1 = Text("4 - 4 + 2 = 2", font_size=42, color=WHITE)
        self.place_in_area(calc1, 'F2', 'F5')

        self.play(Create(verts), Create(edges))
        self.play(FadeIn(face_int), FadeIn(label_f1), FadeIn(face_ext_rect), FadeIn(label_f_ext))
        self.play(Write(calc1))
        self.wait(2)

        # === Animation for Lecture Line 4 ===
        # Add one edge and update formula to 4 - 5 + 3 = 2.
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)

        # Add Diagonal
        diag_edge = Line(v1.get_center(), v3.get_center(), color="#00FFFF")
        
        # New interior faces
        face_a = Polygon(v1.get_center(), v2.get_center(), v3.get_center(), 
                        fill_opacity=0.4, fill_color="#FF00FF", stroke_width=0)
        face_b = Polygon(v1.get_center(), v3.get_center(), v4.get_center(), 
                        fill_opacity=0.4, fill_color="#FF00FF", stroke_width=0)
        
        label_fa = Text("F1", font_size=16, color="#FF00FF")
        label_fb = Text("F2", font_size=16, color="#FF00FF")
        # Fixed (Issue 35): Moved label_fa from C4 to C3 for better centering
        self.place_at_grid(label_fa, 'C3', scale_factor=0.8)
        # Fixed (Issue 33): Moved label_fb from E3 to E4 to avoid overlap with diagonal
        self.place_at_grid(label_fb, 'E4', scale_factor=0.8)
        
        # New Exterior Label
        label_f_ext_new = Text("F3 (Ext)", font_size=18, color="#FF00FF")
        # Fixed (Issue 34): Move to D6 to avoid corner/edge clipping
        self.place_at_grid(label_f_ext_new, 'D6')

        self.play(Create(diag_edge))
        self.play(
            ReplacementTransform(face_int, VGroup(face_a, face_b)),
            ReplacementTransform(label_f1, VGroup(label_fa, label_fb)),
            ReplacementTransform(label_f_ext, label_f_ext_new)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)

        calc2 = Text("4 - 5 + 3 = 2", font_size=42, color=WHITE)
        self.place_in_area(calc2, 'F2', 'F5')
        
        self.play(ReplacementTransform(calc1, calc2))
        self.wait(2)
