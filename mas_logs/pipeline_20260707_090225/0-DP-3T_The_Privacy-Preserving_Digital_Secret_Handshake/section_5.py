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
        self.setup_layout(
            "Phase 3: The Positive Report (Upload)", 
            [
                'If Alice tests positive, she voluntarily shares her seeds.', 
                'She only uploads her seeds for the infection period.', 
                "The server receives seeds, but never Alice's identity.", 
                'This public list contains sick seeds, not contact logs.', 
                "Alice's local contact list remains protected and private."
            ]
        )
        
        # === Animation for Lecture Line 1 ===
        # Alice's phone icon [Asset: phone.svg] turns Red (#EC7063) with a 'Positive' label.
        phone_path = "/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/phone.svg"
        alice_phone_svg = SVGMobject(phone_path, color=WHITE).scale(0.5)
        phone_label = Text("Alice", font_size=14).next_to(alice_phone_svg, UP, buff=0.1)
        alice_phone = VGroup(alice_phone_svg, phone_label)
        self.place_at_grid(alice_phone, 'B2')
        
        positive_label = Text("Positive", font_size=16, color="#EC7063")
        # Issue 42: scale_factor=0.8 at B3
        self.place_at_grid(positive_label, 'B3', scale_factor=0.8)

        self.play(FadeIn(alice_phone))
        self.play(
            alice_phone_svg.animate.set_color("#EC7063").set_fill("#EC7063", opacity=0.8),
            Write(positive_label),
            self.lecture[0].animate.set_color("#EC7063")
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # 14 small Seed icons (#5DADE2) are extracted and labeled 'Infectious Seeds'.
        seeds = VGroup(*[
            Circle(radius=0.08, color="#5DADE2", fill_opacity=1) 
            for _ in range(14)
        ]).arrange_in_grid(rows=2, cols=7, buff=0.1)
        
        seeds_label = Text("Infectious Seeds", font_size=14, color="#5DADE2")
        seeds_container = VGroup(seeds, seeds_label).arrange(DOWN, buff=0.2)
        # Issue 40: move to C2 (to fix vertical crowding)
        self.place_at_grid(seeds_container, 'C2', scale_factor=0.9)

        self.play(
            ReplacementTransform(alice_phone_svg.copy(), seeds),
            FadeIn(seeds_label),
            self.lecture[1].animate.set_color("#5DADE2")
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # The group of Seeds moves into a Cloud [Asset: cloud.svg] icon labeled 'Backend Server' (#F4D336).
        cloud_path = "/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/cloud.svg"
        server_cloud_svg = SVGMobject(cloud_path, color="#F4D336", fill_opacity=1).scale(0.7)
        cloud_text = Text("Backend Server", font_size=12, color=BLACK).move_to(server_cloud_svg.get_center())
        server_cloud = VGroup(server_cloud_svg, cloud_text)
        self.place_at_grid(server_cloud, 'B5')

        self.play(FadeIn(server_cloud))
        self.play(
            seeds_container.animate.move_to(self.grid['B5']).scale(0.3).set_opacity(0),
            self.lecture[2].animate.set_color("#F4D336"),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # The Cloud [Asset: cloud.svg] displays a public list of the 14 Seeds with no personal names.
        list_bg = Rectangle(height=1.5, width=1.2, color=WHITE, fill_opacity=0.1)
        list_title = Text("Public Sick Seeds", font_size=10, color=WHITE).shift(UP*0.5)
        # Representation of seed hashes in the list
        list_entries = VGroup(*[
            Text("0x" + "a1b2..."[i:i+4], font_size=8, color="#5DADE2") 
            for i in range(5)
        ]).arrange(DOWN, buff=0.1).next_to(list_title, DOWN, buff=0.1)
        
        public_list = VGroup(list_bg, list_title, list_entries)
        # Issue 41: move to D5 (visual flow)
        self.place_at_grid(public_list, 'D5')

        self.play(
            FadeIn(public_list),
            self.lecture[3].animate.set_color("#FFFFFF")
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Show a 'Shield' (#58D68D) remaining over Alice's local contact list.
        contact_box = Rectangle(height=0.8, width=1.0, color=GREY, fill_opacity=0.3)
        contact_text = Text("Local Contact Log", font_size=10, color=WHITE)
        local_contacts = VGroup(contact_box, contact_text)
        self.place_at_grid(local_contacts, 'E2')
        
        # Shield icon
        shield_shape = Polygon(
            [0, 0.4, 0], [0.35, 0.2, 0], [0.35, -0.3, 0], 
            [0, -0.5, 0], [-0.35, -0.3, 0], [-0.35, 0.2, 0],
            color="#58D68D", fill_opacity=0.6, stroke_width=2
        )
        shield_label = Text("Private", font_size=10, color=WHITE).move_to(shield_shape.get_center())
        shield_group = VGroup(shield_shape, shield_label)
        self.place_at_grid(shield_group, 'E2', scale_factor=0.8)

        self.play(
            FadeIn(local_contacts),
            FadeIn(shield_group),
            self.lecture[4].animate.set_color("#58D68D")
        )
        self.wait(2)
