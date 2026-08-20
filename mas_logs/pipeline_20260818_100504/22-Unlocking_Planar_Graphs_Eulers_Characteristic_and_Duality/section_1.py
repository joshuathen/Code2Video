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
        lecture_lines = ["Planar graphs have no crossing edges.", 
                         "Identify vertices, edges, and faces.", 
                         "Faces include the infinite outer region.", 
                         "A house graph has five vertices.", 
                         "It has six edges, two faces."]
        self.setup_layout("Prerequisite: Defining the Planar Graph", lecture_lines)
        
        # Build the \"house\" graph
        nodes = {
            "p": Dot(color=WHITE),
            "tl": Dot(color=WHITE),
            "tr": Dot(color=WHITE),
            "bl": Dot(color=WHITE),
            "br": Dot(color=WHITE)
        }
        
        # Use SVG asset
        house_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/house.svg")
        
        pos = {
            "p": "B3",
            "tl": "C2",
            "tr": "C4",
            "bl": "D2",
            "br": "D4"
        }
        for k, v in nodes.items():
            self.place_at_grid(v, pos[k])
        
        edges = VGroup(
            Line(nodes["p"].get_center(), nodes["tl"].get_center(), color=WHITE),
            Line(nodes["p"].get_center(), nodes["tr"].get_center(), color=WHITE),
            Line(nodes["tl"].get_center(), nodes["tr"].get_center(), color=WHITE),
            Line(nodes["tl"].get_center(), nodes["bl"].get_center(), color=WHITE),
            Line(nodes["tr"].get_center(), nodes["br"].get_center(), color=WHITE),
            Line(nodes["bl"].get_center(), nodes["br"].get_center(), color=WHITE)
        )
        
        graph = VGroup(house_asset, edges, *nodes.values())
        self.place_in_area(graph, 'B4', 'E6', scale_factor=0.9)

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(graph))
        self.lecture[0].set_color("#FFFFFF")

        # === Animation for Lecture Line 2 ===
        self.play(nodes["p"].animate.set_color("#FF0000"),
                  nodes["tl"].animate.set_color("#FF0000"),
                  nodes["tr"].animate.set_color("#FF0000"),
                  nodes["bl"].animate.set_color("#FF0000"),
                  nodes["br"].animate.set_color("#FF0000"))
        self.lecture[1].set_color("#FF0000")

        # === Animation for Lecture Line 3 ===
        self.play(edges.animate.set_color("#00FF00"))
        self.lecture[2].set_color("#00FF00")

        # === Animation for Lecture Line 4 ===
        face_rect = Polygon(nodes["tl"].get_center(), nodes["tr"].get_center(), nodes["br"].get_center(), nodes["bl"].get_center(), 
                           fill_opacity=0.3, color="#0000FF", stroke_width=0)
        self.play(FadeIn(face_rect))
        self.lecture[3].set_color("#0000FF")

        # === Animation for Lecture Line 5 ===
        calculation_label = Text("V=5, E=6, F=2", font_size=24, color="#FFFF00")
        self.place_at_grid(calculation_label, 'F5', scale_factor=0.7)
        self.play(Write(calculation_label))
        self.lecture[4].set_color("#FFFF00")
        self.wait(2)
